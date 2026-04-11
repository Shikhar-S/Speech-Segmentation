"""Tests for BuckeyeAlignmentSegment dataclass (no external data)."""

from dataclasses import fields

from src.recipe.segmentation.local.buckeye_data_prep import (
    BuckeyeAlignmentSegment,
)

EXPECTED_FIELDS = (
    "segment_id",
    "speaker_id",
    "track_id",
    "start_time",
    "end_time",
    "text",
    "phones",
    "phone_timestamps",
)


def test_alignment_segment_no_duplicate_fields():
    """All fields are accessible and the dataclass has the expected
    count (bug M37 -- duplicate-field guard)."""
    seg = BuckeyeAlignmentSegment(
        segment_id="s01_track0_0001",
        speaker_id="s01",
        track_id=0,
        start_time=1.0,
        end_time=2.5,
        text="hello world",
        phones=["h", "eh", "l", "ow"],
        phone_timestamps=[(0.0, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.6)],
    )
    dc_fields = fields(seg)
    assert len(dc_fields) == len(EXPECTED_FIELDS)
    # All values round-trip correctly
    assert seg.segment_id == "s01_track0_0001"
    assert seg.speaker_id == "s01"
    assert seg.track_id == 0
    assert seg.start_time == 1.0
    assert seg.end_time == 2.5
    assert seg.text == "hello world"
    assert seg.phones == ["h", "eh", "l", "ow"]
    assert len(seg.phone_timestamps) == 4


def test_alignment_segment_field_order():
    """Field names match expected order."""
    dc_fields = fields(BuckeyeAlignmentSegment)
    actual = tuple(f.name for f in dc_fields)
    assert actual == EXPECTED_FIELDS
