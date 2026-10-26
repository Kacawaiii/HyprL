"""REST contract implementation as an in-process adapter. No socket or server."""

from .model import AlertError


class ReadOnlyAPI:
    def __init__(self, store):
        self.store = store

    def handle(self, method, path, *, authorization="", query=None):
        if method != "GET":
            return 405, {"error": "READ_ONLY"}
        if path != "/v1/alerts":
            return 404, {"error": "NOT_FOUND"}
        query = query or {}
        if set(query) - {"after", "limit"}:
            return 400, {"error": "INVALID_PAGINATION"}
        if (
            not isinstance(authorization, str)
            or not authorization.startswith("Bearer ")
            or len(authorization) > 200
        ):
            return 401, {"error": "UNAUTHORIZED"}
        try:
            items = self.store.feed(
                authorization[7:],
                after=int(query.get("after", "0")),
                limit=int(query.get("limit", "100")),
            )
            return 200, {
                "schema_version": 1,
                "items": items,
                "next_cursor": items[-1]["cursor"] if items else int(query.get("after", "0")),
            }
        except AlertError as error:
            return (
                (401, {"error": "UNAUTHORIZED"})
                if str(error) == "UNAUTHORIZED"
                else (400, {"error": "INVALID_PAGINATION"})
            )
        except (ValueError, TypeError):
            return 400, {"error": "INVALID_PAGINATION"}
