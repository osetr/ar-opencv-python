from classifier import Classifier


model = Classifier()
model.fit("path_to_dir_with_obj", "path_to_dir_without_obj")
model.process_video("video_to_check.mp4", (640, 480), fps=40)
