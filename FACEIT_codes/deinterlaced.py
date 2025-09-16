import subprocess

ffmpeg_exe = r"C:\Users\faezeh.rabbani\Downloads\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
inp = r"C:\Users\faezeh.rabbani\Downloads\1\2.avi"
out = r"C:\Users\faezeh.rabbani\Downloads\1\2_deinterlaced.mp4"

subprocess.run([
    ffmpeg_exe, "-y",
    "-i", inp,
    "-vf", "bwdif=mode=send_field:parity=auto:deint=all",
    "-c:v", "libx264", "-preset", "slow", "-crf", "18",
    "-c:a", "copy",
    out
], check=True)
