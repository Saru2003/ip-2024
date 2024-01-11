from moviepy.editor import *
audiopath="recording1.wav"
videopath="animation_output2.mp4"
vid_clip=VideoFileClip(videopath)
aud_clip=AudioFileClip(audiopath)

merge=vid_clip.set_audio(aud_clip)
merge.write_videofile('test_merge.mp4')

