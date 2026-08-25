from pydantic import BaseModel, ConfigDict


class StrictBaseModel(BaseModel):
    """Base for every pipeline config class -- rejects unrecognized fields
    instead of silently dropping them, e.g. a key nested under the wrong
    section."""

    model_config = ConfigDict(extra="forbid")
