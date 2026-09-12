import pytest

from brl.cvr.ids import ChannelId, PlanNodeId, StationId, require_channel, require_station


def test_ids_with_same_integer_are_not_equal():
    assert ChannelId(1) != StationId(1)
    assert PlanNodeId("fixed-1") != PlanNodeId("fixed-2")


def test_runtime_guards_reject_the_wrong_identifier_kind():
    with pytest.raises(TypeError, match="StationId"):
        require_station(ChannelId(1))
    with pytest.raises(TypeError, match="ChannelId"):
        require_channel(StationId(1))
