from os import getcwd
import httpx
import numpy as np
from numpy.f2py.auxfuncs import throw_error
from telegram import Update, Sticker
from telegram.ext import Updater, MessageHandler, CallbackContext, Filters
from concurrent.futures import ThreadPoolExecutor, wait
import cv2
import imageio
import os

BOT_TOKEN = "<BOT_TOKENS>"

# 在目录下创建你的token文件
if not os.path.exists("MYTOKEN"):
    throw_error("创建你的TOKEN文件 [MYTOKEN] , 内容就写你的机器人TOKEN!")

with open("MYTOKEN") as f:
    BOT_TOKEN = f.readline()

# 贴纸存储路径
SAVE_DIR = "stickers"

# 创建文件夹
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 创建线程池
executor = ThreadPoolExecutor(max_workers=16)  # 最大并发线程数，可以根据需要调整

def webm_binary_to_gif_with_transparency(webm_binary_data, output_gif_path,fps=30):
    # 获取当前目录
    webm = os.path.join(getcwd(), "{}.webm".format(output_gif_path))
    gif = os.path.join(getcwd(), "{}.gif".format(output_gif_path))

    if not os.path.exists(webm) or os.path.getsize(webm) == 0:
        with open(webm, 'wb') as f:
            f.write(webm_binary_data)

    # 使用 OpenCV 读取 WebM 文件
    cap = cv2.VideoCapture(webm)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second (FPS): {fps}")

    # 获取视频的帧数
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV 默认是 BGR 格式，需要转换成 RGBA（带透明度）
        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 添加到帧列表
        frames.append(rgba_frame)

    # 使用 ImageIO 保存为 GIF，带透明度
    with imageio.get_writer(gif, mode='I', fps=fps, loop=0, palettesize=256) as writer:
        for frame in frames:
            # 将每一帧转换为 uint8 类型（imageio 需要）
            frame = np.uint8(frame)
            writer.append_data(frame)
    cap.release()
    return True

def convert_webp2png(webpImageBuffer):
    image = np.frombuffer(webpImageBuffer, np.uint8)
    imageDecode = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    if imageDecode is None:
        return

    _, pngImage = cv2.imencode('.png', imageDecode)
    return pngImage.tobytes()

def download_sticker(context, file_id, localFilePath, is_animated, is_video):
    """下载并保存贴纸"""
    file = context.bot.get_file(file_id)
    img = file.file_path

    needLoad = False
    if is_video:
        if not os.path.exists(localFilePath+".webm") or os.path.getsize(localFilePath+".webm") == 0:
            needLoad = True
    else:
        needLoad = True

    extName = ".png"
    pngImageBytes = None

    if needLoad:
        blob = httpx.get(img)

        if blob.status_code != 200:
            print(f"请求错误，文件 ID: {file_id}")
            return
        pngImageBytes = blob.content

    if is_video:
        extName = ".webm"
        webm_binary_to_gif_with_transparency(pngImageBytes, localFilePath)
        return
    elif is_animated:
        extName = ".jpg"
    else:
        pngImageBytes = convert_webp2png(pngImageBytes)
    with open(localFilePath+extName, "wb") as f:
        f.write(pngImageBytes)
        f.flush()

def sticker_handler(update: Update, context: CallbackContext):
    """处理贴纸并保存，同时获取贴纸包及其所有贴纸"""
    sticker = update.message.sticker
    set_name = sticker.set_name  # 贴纸包名称（如果有）
    stickerSize = -1

    if set_name:
        # 获取贴纸包信息
        sticker_set = context.bot._post('getStickerSet', {'name': set_name})
        name = sticker_set['name']
        stickers = sticker_set['stickers']
        stickerSize = len(stickers)

        # 根据贴纸包名称创建文件夹
        tgt = os.path.join(SAVE_DIR, name)
        if not os.path.exists(tgt):
            os.makedirs(tgt)

        # 使用线程池并发下载所有贴纸
        futures = []  # 用来存储所有的 Future 对象
        for sticker in stickers:
            sti = Sticker.de_json(sticker, context.bot)
            is_animated = sti.is_animated
            is_video = sti.is_video
            file_id = sti.file_id
            file_unique_id = sti.file_unique_id
            emoji = sti.emoji
            print(file_id, file_unique_id, is_animated, is_video)

            downloaded_file = os.path.join(tgt, f"{emoji}_{file_unique_id}")

            # 启动一个线程来处理每个贴纸的下载任务，并将返回的 Future 对象加入 futures 列表
            future = executor.submit(download_sticker, context, file_id, downloaded_file, is_animated, is_video)
            futures.append(future)

        # 等待所有下载任务完成
        wait(futures)

    context.bot.send_message(
        update.message.chat_id,
        f"下载 Sticker Pack {set_name} [{stickerSize}个] OK!",
        entities=update.message.entities
    )

async def error_handler(update: Update, context: CallbackContext):
    """错误处理"""
    print(f"错误: {context.error}")

def main():
    """主函数"""
    app = Updater(BOT_TOKEN)
    dispatcher = app.dispatcher

    # 处理贴纸消息
    dispatcher.add_handler(MessageHandler(~Filters.command, sticker_handler))

    # 处理错误
    dispatcher.add_error_handler(error_handler)

    # 启动机器人
    app.start_polling()
    print("Bot 启动.")
    app.idle()

if __name__ == "__main__":
    main()
