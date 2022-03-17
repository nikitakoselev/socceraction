# -*- coding: utf-8 -*-
"""Kloppy EventDataset to SPADL converter."""
from typing import Dict, List, Optional, Tuple

import pandas as pd  # type: ignore
from pandera.typing import DataFrame

from . import config as spadlconfig
from .base import _add_dribbles
from .schema import SPADLSchema

_has_kloppy = True
try:
    from kloppy.domain import (
        BodyPart,
        CarryEvent,
        CoordinateSystem,
        Dimension,
        Event,
        EventDataset,
        EventType,
        FoulCommittedEvent,
        Orientation,
        Origin,
        PassEvent,
        PassResult,
        PassType,
        PitchDimensions,
        Provider,
        RecoveryEvent,
        SetPieceType,
        ShotEvent,
        ShotResult,
        TakeOnEvent,
        TakeOnResult,
        VerticalOrientation,
    )
except ImportError:
    _has_kloppy = False


def convert_to_actions(dataset: EventDataset, home_team_id: int) -> DataFrame[SPADLSchema]:
    """
    Convert a Kloppy event data set to SPADL actions.

    Parameters
    ----------
    dataset : EventDataset
        A Kloppy event data set.
    home_team_id : int
        ID of the home team in the corresponding game.

    Raises
    ------
    ImportError
        If the Kloppy package is not installed.

    Returns
    -------
    actions : pd.DataFrame
        DataFrame with corresponding SPADL actions.

    """
    if not _has_kloppy:
        raise ImportError("Kloppy is required. Install with `pip install kloppy`.")

    new_dataset = dataset.transform(
        to_orientation=Orientation.FIXED_HOME_AWAY,
        to_coordinate_system=_SoccerActionCoordinateSystem(normalized=False),
    )

    actions = []
    for event in new_dataset.events:
        action = dict(
            game_id=0,  # TDDO: get the match id
            original_event_id=event.event_id,
            period_id=event.period.id,
            time_seconds=event.timestamp,
            team_id=event.team.team_id if event.team else None,
            player_id=event.player.player_id if event.player else None,
            start_x=event.coordinates.x if event.coordinates else None,
            start_y=event.coordinates.y if event.coordinates else None,
            **_get_end_location(event),
            **_parse_event(event),
        )
        actions.append(action)

    actions = (
        pd.DataFrame(actions)
        .sort_values(["game_id", "period_id", "time_seconds"])
        .reset_index(drop=True)
    )

    actions = actions[actions.type_id != spadlconfig.actiontypes.index("non_action")]
    actions["action_id"] = range(len(actions))
    actions = _add_dribbles(actions)
    return actions.pipe(DataFrame[SPADLSchema])


class _SoccerActionCoordinateSystem(CoordinateSystem):
    @property
    def provider(self) -> Provider:
        return "SoccerAction"

    @property
    def origin(self) -> Origin:
        return Origin.BOTTOM_LEFT

    @property
    def vertical_orientation(self) -> VerticalOrientation:
        return VerticalOrientation.BOTTOM_TO_TOP

    @property
    def pitch_dimensions(self) -> PitchDimensions:
        return PitchDimensions(
            x_dim=Dimension(0, spadlconfig.field_length),
            y_dim=Dimension(0, spadlconfig.field_width),
        )


def _get_end_location(event: Event) -> Dict[str, Optional[float]]:
    if isinstance(event, PassEvent) and event.result == PassResult.COMPLETE:
        if event.receiver_coordinates:
            return {
                "end_x": event.receiver_coordinates.x,
                "end_y": event.receiver_coordinates.y,
            }
    elif isinstance(event, CarryEvent):
        if event.end_coordinates:
            return {
                "end_x": event.end_coordinates.x,
                "end_y": event.end_coordinates.y,
            }
    elif isinstance(event, ShotEvent):
        if event.result_coordinates:
            return {
                "end_x": event.result_coordinates.x,
                "end_y": event.result_coordinates.y,
            }
    if event.coordinates:
        return {"end_x": event.coordinates.x, "end_y": event.coordinates.y}
    return {"end_x": None, "end_y": None}


def _parse_event(event: Event) -> Dict[str, int]:
    events = {
        EventType.GENERIC: _parse_event_as_non_action,
        EventType.PASS: _parse_pass_event,
        EventType.SHOT: _parse_shot_event,
        EventType.TAKE_ON: _parse_dribble_event,
        EventType.CARRY: _parse_carry_event,
        EventType.RECOVERY: _parse_interception_event,  # TODO: result
        # EventType.FOUL_COMMITTED: _parse_event_as_non_action,
        # missing on-ball events
        # - clearance
        # - bad touch
        # - tackle
        # - keeper_save
        # - keeper_claim
        # - keeper_punch
        # other non-action events
        # EventType.SUBSTITUTION: _parse_event_as_non_action,
        # EventType.CARD: _parse_event_as_non_action,
        # EventType.PLAYER_ON: _parse_event_as_non_action,
        # EventType.PLAYER_OFF: _parse_event_as_non_action,
        # EventType.BALL_OUT: _parse_event_as_non_action,
        # EventType.FORMATION_CHANGE:_parse_event_as_non_action,
    }
    parser = events.get(event.event_type, _parse_event_as_non_action)
    a, r, b = parser(event)
    return {
        "type_id": spadlconfig.actiontypes.index(a),
        "result_id": spadlconfig.results.index(r),
        "bodypart_id": spadlconfig.bodyparts.index(b),
    }


def _qualifiers(event: Event) -> List:
    if event.qualifiers:
        return [q.value for q in event.qualifiers]
    return []


def _parse_event_as_non_action(event: Event) -> Tuple[str, str, str]:
    a = "non_action"
    r = "success"
    b = "foot"
    return a, r, b


def _parse_pass_event(event: PassEvent) -> Tuple[str, str, str]:  # noqa: C901

    qualifiers = _qualifiers(event)

    a = "pass"  # default
    if SetPieceType.FREE_KICK in qualifiers:
        if (
            PassType.CHIPPED_PASS in qualifiers
            or PassType.CROSS in qualifiers
            or PassType.HIGH_PASS in qualifiers
        ):
            a = "freekick_crossed"
        else:
            a = "freekick_short"
    elif SetPieceType.CORNER_KICK in qualifiers:
        if (
            PassType.CHIPPED_PASS in qualifiers
            or PassType.CROSS in qualifiers
            or PassType.HIGH_PASS in qualifiers
        ):
            a = "corner_crossed"
        else:
            a = "corner_short"
    elif SetPieceType.GOAL_KICK in qualifiers:
        a = "goalkick"
    elif SetPieceType.THROW_IN in qualifiers:
        a = "throw_in"
    elif PassType.CROSS in qualifiers:
        a = "cross"
    else:
        a = "pass"

    if event.result in [PassResult.INCOMPLETE, PassResult.OUT]:
        r = "fail"
    elif event.result == PassResult.OFFSIDE:
        r = "offside"
    else:
        r = "success"

    if BodyPart.HEAD in qualifiers:
        b = "head"
    elif BodyPart.RIGHT_FOOT in qualifiers or BodyPart.LEFT_FOOT in qualifiers:
        b = "foot"
    elif BodyPart.CHEST in qualifiers or BodyPart.OTHER in qualifiers:
        b = "other"
    else:
        b = "foot"

    return a, r, b


def _parse_shot_event(event: ShotEvent) -> Tuple[str, str, str]:
    qualifiers = _qualifiers(event)

    if SetPieceType.FREE_KICK in qualifiers:
        a = "shot_freekick"
    elif SetPieceType.PENALTY in qualifiers:
        a = "shot_penalty"
    else:
        a = "shot"

    if event.result == ShotResult.GOAL:
        r = "success"
    elif event.result == ShotResult.OWN_GOAL:
        a = "bad_touch"
        r = "own_goal"
    else:
        r = "fail"

    if BodyPart.HEAD in qualifiers:
        b = "head"
    elif (
        BodyPart.RIGHT_FOOT in qualifiers
        or BodyPart.LEFT_FOOT in qualifiers
        or BodyPart.DROP_KICK in qualifiers
    ):
        b = "foot"
    else:
        b = "other"

    return a, r, b


def _parse_dribble_event(event: TakeOnEvent) -> Tuple[str, str, str]:
    a = "take_on"

    if event.result == TakeOnResult.COMPLETE:
        r = "success"
    else:
        r = "fail"

    b = "foot"

    return a, r, b


def _parse_carry_event(_e: CarryEvent) -> Tuple[str, str, str]:
    a = "dribble"
    r = "success"
    b = "foot"
    return a, r, b


def _parse_interception_event(event: RecoveryEvent) -> Tuple[str, str, str]:
    a = "interception"
    r = "success"  # not implemented in Kloppy
    b = "foot"
    return a, r, b


def _parse_foul_event(event: FoulCommittedEvent) -> Tuple[str, str, str]:
    a = "foul"
    r = "success"  # seperate event in Kloppy
    b = "foot"

    return a, r, b
