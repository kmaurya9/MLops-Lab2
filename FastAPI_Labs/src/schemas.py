from pydantic import BaseModel, ConfigDict, Field

EXAMPLE_SAMPLE = {
    "alcohol": 13.2,
    "malic_acid": 1.78,
    "ash": 2.14,
    "alcalinity_of_ash": 11.2,
    "magnesium": 100.0,
    "total_phenols": 2.65,
    "flavanoids": 2.76,
    "nonflavanoid_phenols": 0.26,
    "proanthocyanins": 1.28,
    "color_intensity": 4.38,
    "hue": 1.05,
    "od280_od315_ratio": 3.4,
    "proline": 1050.0,
}


class WineSample(BaseModel):
    """
    Chemical analysis of a single wine. Bounds are loose sanity limits around
    the ranges seen in the UCI Wine dataset, not hard chemistry rules.
    """

    model_config = ConfigDict(json_schema_extra={"examples": [EXAMPLE_SAMPLE]})

    alcohol: float = Field(..., ge=8, le=18, description="Alcohol, % by volume")
    malic_acid: float = Field(..., ge=0, le=10)
    ash: float = Field(..., ge=0, le=5)
    alcalinity_of_ash: float = Field(..., ge=0, le=40)
    magnesium: float = Field(..., ge=50, le=200)
    total_phenols: float = Field(..., ge=0, le=5)
    flavanoids: float = Field(..., ge=0, le=6)
    nonflavanoid_phenols: float = Field(..., ge=0, le=1)
    proanthocyanins: float = Field(..., ge=0, le=5)
    color_intensity: float = Field(..., ge=0, le=15)
    hue: float = Field(..., ge=0, le=2)
    od280_od315_ratio: float = Field(..., ge=0, le=5, description="OD280/OD315 of diluted wine")
    proline: float = Field(..., ge=200, le=2000)


class BatchRequest(BaseModel):
    samples: list[WineSample] = Field(..., min_length=1, max_length=100)


class WinePrediction(BaseModel):
    class_id: int
    cultivar: str
    confidence: float
    probabilities: dict[str, float]


class BatchPrediction(BaseModel):
    count: int
    predictions: list[WinePrediction]


class ModelMetadata(BaseModel):
    model: str
    trained_at: str
    features: list[str]
    classes: list[str]
    metrics: dict
