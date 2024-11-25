from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Emotion Analysis Server is running"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # 이미지 읽기
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # 이미지 처리 (예: 크기 조정, Grayscale 변환)
        image = image.resize((128, 128)).convert("L")
        
        # 감정 분석 결과 (임시, 실제 모델 연결 필요)
        dummy_result = {
            "emotion": "happy",
            "confidence": 0.95
        }
        
        return JSONResponse(content={"status": "success", "result": dummy_result})
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
