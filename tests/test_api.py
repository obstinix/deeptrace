import sys
from io import BytesIO

from PIL import Image

sys.path.insert(0, "src")

def _jpeg(size=(64,64)):
    buf = BytesIO()
    Image.new("RGB", size, (100,150,200)).save(buf, format="JPEG")
    return buf.getvalue()

def test_health():
    from api.main import app
    from fastapi.testclient import TestClient
    r = TestClient(app).get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

def test_predict_no_model_503():
    from api.main import app
    from fastapi.testclient import TestClient
    app.state.predictor = None
    r = TestClient(app).post("/api/predict/image",
                              files={"file":("t.jpg",_jpeg(),"image/jpeg")})
    assert r.status_code == 503

def test_predict_too_large_413():
    from fastapi.testclient import TestClient
    from api.main import app
    app.state.predictor = object()
    r = TestClient(app).post("/api/predict/image",
                              files={"file":("big.jpg",b"x"*(11*1024*1024),"image/jpeg")})
    assert r.status_code == 413

def test_predict_wrong_type_415():
    from fastapi.testclient import TestClient
    from api.main import app
    app.state.predictor = object()
    r = TestClient(app).post("/api/predict/image",
                              files={"file":("f.pdf",b"%PDF","application/pdf")})
    assert r.status_code == 415
