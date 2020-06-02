from mlpm.solver import Solver
import imageio
from first_order_motion.utility import load_checkpoints, make_animation, find_best_frame  
from skimage.transform import resize
import uuid
import os
from skimage import img_as_ubyte
class VideoConverterSolver(Solver):
    def __init__(self, toml_file=None):
        super().__init__(toml_file)
        # Do you Init Work here
        self.classifer = None
        self.generator, self.kp_detector = load_checkpoints(config_path='first_order_motion/config/vox-256.yaml', checkpoint_path='pretrained/vox-cpk.pth.tar', cpu=True)
        self.ready()
        
    def infer(self, data):
        # if you need to get file uploaded, get the path from input_file_path in data
        video = data['input_file_path']
        source_image = 'first_order_motion/images/monalisa.jpg'
        source_image = imageio.imread(source_image)
        reader = imageio.get_reader(video)
        fps = reader.get_meta_data()['fps']
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()
        source_image = resize(source_image, (256, 256))[..., :3]
        driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
        predictions = make_animation(source_image, driving_video, self.generator, self.kp_detector, relative=True, adapt_movement_scale=True, cpu=True)
        fileid = str(uuid.uuid4())+".mp4"
        target_path = os.path.join('.','static')
        if not os.path.isdir(target_path):
            os.makedirs(target_path)
        result_path = os.path.join('.','static',fileid)
        imageio.mimsave(result_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)


        return {"output":result_path} # return a dict