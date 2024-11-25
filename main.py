from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from model import process_image, predict_emotion
import os
import tempfile

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Emotion Analysis Server is running"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # 임시 파일 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(await file.read())
            tmp_path = tmp_file.name

        # 이미지 전처리 및 추론
        image_tensor = process_image(tmp_path)
        result = predict_emotion(image_tensor)

        # 임시 파일 삭제
        os.remove(tmp_path)

        # 결과 반환
        return JSONResponse(content={"status": "success", "result": result})
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
