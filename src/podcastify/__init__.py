"""Podcastify: turn a PDF, blog post or YouTube video into a two-host podcast."""

__version__ = "0.1.0"


def _use_os_certificates() -> None:
    """Verify HTTPS with the operating system's certificate store.

    On Windows, Python's bundled view of the cert store can miss root certificates
    that Windows only fetches on demand, which breaks model downloads with
    CERTIFICATE_VERIFY_FAILED. truststore makes Python ask Windows directly, like
    a browser does (and it respects corporate proxies / antivirus certificates).
    """
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:  # pragma: no cover - never block startup on this
        pass


_use_os_certificates()
