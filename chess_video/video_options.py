def validate_video_options(width: int, height: int, fps: int) -> None:
    if (
        any(type(value) is not int for value in (width, height, fps))
        or not 240 <= width < height <= 3840
        or width % 2
        or height % 2
        or height * 20 < width * 27
        or not 1 <= fps <= 120
    ):
        raise ValueError(
            "Output requires even integer dimensions with 240 <= width < height <= 3840, "
            "height >= width * 1.35, and integer fps between 1 and 120"
        )
