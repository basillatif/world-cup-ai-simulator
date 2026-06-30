from src.app.navigation import ARCHIVE_PAGES


def test_navigation_paths_are_relative_to_streamlit_entrypoint():
    assert all(path.startswith("pages/") for _, path in ARCHIVE_PAGES)
    assert not any(path.startswith("src/app/") for _, path in ARCHIVE_PAGES)
    assert not any("Archive" in label or "Archive" in path for label, path in ARCHIVE_PAGES)
