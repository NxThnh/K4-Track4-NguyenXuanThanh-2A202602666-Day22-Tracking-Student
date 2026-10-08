"""Unit tests cho các hàm phụ trợ tracking mà không cần GPU hoặc model."""

import pytest
from run_tracking import color_for_id


def test_color_for_id_deterministic() -> None:
    """Kiểm tra color_for_id trả về cùng một màu khi cùng ID."""
    color_1 = color_for_id(42)
    color_2 = color_for_id(42)
    assert color_1 == color_2


def test_color_for_id_range() -> None:
    """Kiểm tra các giá trị kênh màu nằm trong khoảng 64 đến 254."""
    for track_id in [1, 2, 100, 999]:
        color = color_for_id(track_id)
        assert len(color) == 3
        for channel in color:
            assert isinstance(channel, int)
            assert 64 <= channel <= 254


def test_color_for_id_different_ids() -> None:
    """Kiểm tra hai ID khác nhau sinh ra màu khác nhau."""
    assert color_for_id(1) != color_for_id(2)
