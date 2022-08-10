import json
from dataset.dataset_interface import NerfDataset
from utils.image_utils import *


class NerfingMVSDataset(NerfDataset):
    def __init__(self, basedir, **kwargs):
        super().__init__("nerfingmvs", **kwargs)
        self.scene_name = basedir.split("/")[-1]
        if kwargs.get("load_depth_range_from_file", False):
            with open(os.path.join(basedir, 'min_max_depth.json'), 'r') as fp:
                f = json.load(fp)
                self.near = f["min_depth"] * 0.9
                self.far = f["max_depth"] * 1.1
            print("LOAD FROM FILE!!!!!!!!!!!!!!!!!!!!!!!")
            print(self.near)
            print(self.far)

        if self.load_priors:
            with open(os.path.join(basedir, 'avg_irradiance.json'), 'r') as fp:
                f = json.load(fp)
                self.prior_irradiance_mean = f["mean_" + self.prior_type]

        with open(os.path.join(basedir, 'transform.json'.format(self.split)), 'r') as fp:
            self.meta = json.load(fp)

        self.instance_color_list = []
        self.instance_num = 0

        self.basedir = basedir

        self.skip = kwargs.get("skip", 1)
        if self.split == "train":
            self.skip = 1

        self.camera_angle_x = float(self.meta['camera_angle_x'])

        self.original_height = self.meta["h"]
        self.original_width = self.meta["w"]

        self.height = int(self.original_height * self.scale)
        self.width = int(self.original_width * self.scale)
        self.focal = .5 * self.width / np.tan(0.5 * self.camera_angle_x)

        self.image_list = []
        image_list_file = open(os.path.join(basedir, "{}.txt".format(self.split)))
        while True:
            line = image_list_file.readline()
            if not line:
                break
            self.image_list.append(line.strip())

        self.frames = self.meta['frames']
        self.frame_image_list = [os.path.split(self.frames[i]['file_path'])[-1] for i in range(len(self.frames))]

    # 	self.load_near_far_plane()
    #
    # def load_near_far_plane(self):
    # 	"""
    # 	Load near and far plane
    # 	:return:
    # 	"""
    # 	self.near = 1
    # 	self.far = 20

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        sample = {}
        frame_index = self.frame_image_list.index(self.image_list[index])
        frame = self.meta['frames'][::self.skip][self.image_list[frame_index]]
        image_file_path = os.path.join(self.basedir, "images", self.image_list[index])
        prior_albedo_file_path = os.path.join(self.basedir, "images",
                                              "{}_{}_r.png".format(self.image_list[index], self.prior_type))
        prior_irradiance_file_path = os.path.join(self.basedir, "images",
                                                  "{}_{}_s.png".format(self.image_list[index], self.prior_type))

        # (1) load RGB Image
        if self.load_image:
            sample["image"] = load_image_from_path(image_file_path, scale=self.scale, is_colmap=True)
        if self.load_normal:
            raise ValueError
        if self.load_albedo:
            raise ValueError
        if self.load_roughness:
            raise ValueError
        if self.load_depth:
            raise ValueError
        if self.load_irradiance:
            raise ValueError
        if self.load_diffuse_specular:
            raise ValueError
        if self.load_priors:
            sample["prior_albedo"] = load_image_from_path(prior_albedo_file_path, scale=self.scale)
            sample["prior_irradiance"] = load_image_from_path(prior_irradiance_file_path, scale=self.scale)

        # (2) load instance_label_mask
        if self.load_instance_label_mask:
            raise ValueError

        # (3) load pose information
        pose = np.array(frame['transform_matrix']).astype(np.float32)
        # Mitsuba --> camera forward is +Z !!
        # pose[:3, 0] *= -1
        # pose[:3, 2] *= -1
        sample["pose"] = pose
        return sample

    def get_test_render_poses(self):
        # TODO : implement
        return None
