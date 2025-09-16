import subprocess

input_file = r"C:\Users\faezeh.rabbani\Downloads\1\2.avi"
output_file = r"C:\Users\faezeh.rabbani\Downloads\1\2_deinterlaced.mp4"

command = [
    "ffmpeg",
    "-i", input_file,
    "-vf", "yadif",          # apply deinterlacing
    "-c:v", "libx264",       # H.264 codec
    "-preset", "fast",
    "-crf", "22",
    "-c:a", "aac",
    "-b:a", "128k",
    output_file
]

subprocess.run(command, check=True)
print(f"Deinterlaced video saved at: {output_file}")
