import cv2
from ultralytics import YOLO
import RPi.GPIO as GPIO
import time

model = YOLO('FaceDetectFinal.pt')  # 載入 YOLOv8 模型
target_classes = ['CHEN', 'Amy', 'Ben', 'KengLiHU', 'Thompson']  # 目標人物清單
conf_threshold = 0.7  # 設定置信度閾值

# 初始化設備
cap = cv2.VideoCapture(0)  
gpio_initialized = False # 紀錄 GPIO 是否初始化
print("System ready and camera initialized.")


try:
    while True:
        # 讀取影像
        ret, frame = cap.read() 
        if not ret:
            print("Failed to capture image")
            break

        # 使用 YOLOv8 模型進行偵測
        results = model.predict(frame, conf=conf_threshold, verbose=False)

        # 檢查是否偵測到指定人物
        detected = False
        for result in results:  # 迭代每個偵測結果
            for box in result.boxes:  
                cls = int(box.cls[0])  
                class_name = model.names[cls]  
                if class_name in target_classes:
                    detected = True
                    print(f"Detect {class_name}")
                    break  
            if detected: # 偵測到一個人就該開門了，不用繼續檢查
                break

        # 控制開關
        if detected:
            if not gpio_initialized:  
                switch_pin = 23 
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(switch_pin, GPIO.OUT)  
                gpio_initialized = True  # 標誌設為已初始化
            GPIO.output(switch_pin, GPIO.LOW)  # 低電位打開繼電器開關
            time.sleep(1)
            print("Target detected, switch turned ON.")
        else:
            if gpio_initialized: 
                GPIO.cleanup()  
                gpio_initialized = False  # 標誌設為未初始化
            print("Target not detected, switch turned OFF.")

except KeyboardInterrupt: # Ctrl+C 中斷程式
    print("Process interrupted by user.")
finally:
    # 清理資源
    cap.release()
    cv2.destroyAllWindows() 
    if gpio_initialized:  
        GPIO.cleanup()
    print("Resources released and GPIO cleaned up.")