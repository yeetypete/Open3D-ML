import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont
import numpy.typing as npt
from matplotlib import font_manager

class BoundingBox3D:
    """Class that defines an axially-oriented bounding box."""

    next_id = 1

    def __init__(self,
                center,
                front,
                up,
                left,
                size,
                label_class,
                confidence,
                meta=None,
                show_class=False,
                show_confidence=False,
                show_meta=None,
                identifier=None,
                arrow_length=1.0,
                center_cov=np.zeros((3,3))):
        """Creates a bounding box.

        Front, up, left define the axis of the box and must be normalized and
        mutually orthogonal.

        Args:
            center: (x, y, z) that defines the center of the box.
            front: normalized (i, j, k) that defines the front direction of the
                box.
            up: normalized (i, j, k) that defines the up direction of the box.
            left: normalized (i, j, k) that defines the left direction of the
                box.
            size: (width, height, depth) that defines the size of the box, as
                measured from edge to edge.
            label_class: integer specifying the classification label. If an LUT
                is specified in create_lines() this will be used to determine
                the color of the box.
            confidence: confidence level of the box.
            meta: a user-defined string (optional).
            show_class: displays the class label in text near the box
                (optional).
            show_confidence: displays the confidence value in text near the box
                (optional).
            show_meta: displays the meta string in text near the box (optional).
            identifier: a unique integer that defines the id for the box
                (optional, will be generated if not provided).
            arrow_length: the length of the arrow in the front_direct. Set to
                zero to disable the arrow (optional).
        """
        assert (len(center) == 3)
        assert (len(front) == 3)
        assert (len(up) == 3)
        assert (len(left) == 3)
        assert (len(size) == 3)
        assert (center_cov.shape == (3,3))

        self.center = np.array(center, dtype="float32")
        self.front = np.array(front, dtype="float32")
        self.up = np.array(up, dtype="float32")
        self.left = np.array(left, dtype="float32")
        self.size = size
        self.label_class = label_class
        self.confidence = confidence
        self.meta = meta
        self.show_class = show_class
        self.show_confidence = show_confidence
        self.show_meta = show_meta
        self.center_cov = center_cov

        if identifier is not None:
            self.identifier = identifier
        else:
            self.identifier = "box:" + str(BoundingBox3D.next_id)
            BoundingBox3D.next_id += 1
        self.arrow_length = arrow_length

    def __repr__(self):
        s = str(self.identifier) + " (class=" + str(
            self.label_class) + ", conf=" + str(self.confidence)
        if self.meta is not None:
            s = s + ", meta=" + str(self.meta)
        s = s + ")"
        return s

    @staticmethod
    def create_lines(boxes, lut=None, out_format="lineset"):
        """Creates a LineSet that can be used to render the boxes.

        Args:
            boxes: the list of bounding boxes
            lut: a ml3d.vis.LabelLUT that is used to look up the color based on
                the label_class argument of the BoundingBox3D constructor. If
                not provided, a color of 50% grey will be used. (optional)
            out_format (str): Output format. Can be "lineset" (default) for the
                Open3D lineset or "dict" for a dictionary of lineset properties.

        Returns:
            For out_format == "lineset": open3d.geometry.LineSet
            For out_format == "dict": Dictionary of lineset properties
                ("vertex_positions", "line_indices", "line_colors", "bbox_labels",
                "bbox_confidences").
        """
        if out_format not in ('lineset', 'dict'):
            raise ValueError("Please specify an output_format of 'lineset' "
                             "(default) or 'dict'.")

        nverts = 14
        nlines = 17
        points = np.zeros((nverts * len(boxes), 3), dtype="float32")
        indices = np.zeros((nlines * len(boxes), 2), dtype="int32")
        colors = np.zeros((nlines * len(boxes), 3), dtype="float32")

        for i, box in enumerate(boxes):
            if np.allclose(box.size, 0) and np.allclose(box.arrow_length, 0):
                continue
            pidx = nverts * i
            x = 0.5 * box.size[0] * box.left
            y = 0.5 * box.size[1] * box.up
            z = 0.5 * box.size[2] * box.front
            arrow_tip = box.center + z + box.arrow_length * box.front
            arrow_mid = box.center + z + 0.60 * box.arrow_length * box.front
            head_length = 0.3 * box.arrow_length
            # It seems to be substantially faster to assign directly for the
            # points, as opposed to points[pidx:pidx+nverts] = np.stack((...))
            points[pidx] = box.center + x + y + z
            points[pidx + 1] = box.center - x + y + z
            points[pidx + 2] = box.center - x + y - z
            points[pidx + 3] = box.center + x + y - z
            points[pidx + 4] = box.center + x - y + z
            points[pidx + 5] = box.center - x - y + z
            points[pidx + 6] = box.center - x - y - z
            points[pidx + 7] = box.center + x - y - z
            points[pidx + 8] = box.center + z
            points[pidx + 9] = arrow_tip
            points[pidx + 10] = arrow_mid + head_length * box.up
            points[pidx + 11] = arrow_mid - head_length * box.up
            points[pidx + 12] = arrow_mid + head_length * box.left
            points[pidx + 13] = arrow_mid - head_length * box.left

        # It is faster to break the indices and colors into their own loop.
        for i, box in enumerate(boxes):
            pidx = nverts * i
            idx = nlines * i
            indices[idx:idx +
                    nlines] = ((pidx, pidx + 1), (pidx + 1, pidx + 2),
                               (pidx + 2, pidx + 3), (pidx + 3, pidx),
                               (pidx + 4, pidx + 5), (pidx + 5, pidx + 6),
                               (pidx + 6, pidx + 7), (pidx + 7, pidx + 4),
                               (pidx + 0, pidx + 4), (pidx + 1, pidx + 5),
                               (pidx + 2, pidx + 6), (pidx + 3, pidx + 7),
                               (pidx + 8, pidx + 9), (pidx + 9, pidx + 10),
                               (pidx + 9, pidx + 11), (pidx + 9,
                                                       pidx + 12), (pidx + 9,
                                                                    pidx + 13))

            if lut is not None and box.label_class in lut.labels:
                label = lut.labels[box.label_class]
                c = (label.color[0], label.color[1], label.color[2])
            else:
                if box.confidence == -1.0:
                    c = (0., 1.0, 0.)  # GT: Green
                elif box.confidence >= 0 and box.confidence <= 1.0:
                    c = (1.0, 0., 0.)  # Prediction: red
                else:
                    c = (0.5, 0.5, 0.5)  # Grey

                if hasattr(box, 'is_fusion') and box.is_fusion:
                    c = (1.0, 0.5, 0.0) # orange

            colors[idx:idx +
                   nlines] = c  # copies c to each element in the range
        if out_format == "lineset":
            lines = o3d.geometry.LineSet()
            lines.points = o3d.utility.Vector3dVector(points)
            lines.lines = o3d.utility.Vector2iVector(indices)
            lines.colors = o3d.utility.Vector3dVector(colors)
        elif out_format == "dict":
            lines = {
                "vertex_positions": points,
                "line_indices": indices,
                "line_colors": colors,
                "bbox_labels": tuple(b.label_class for b in boxes),
                "bbox_confidences": tuple(b.confidence for b in boxes)
            }

        return lines
    
    @staticmethod
    def create_meshes(boxes, lut=None, n_std=1.0, resolution=20, reg=1e-6, axis_min=0.0):
        """Creates an ellipsoid covariance meshes for the center of each box."""
        def is_valid_cov(cov: npt.NDArray, reg=1e-6) -> bool:
            """Check if covariance matrix is positive semi-definite and symmetric"""
            cov = cov + reg * np.eye(cov.shape[0])
            if not np.allclose(cov, cov.T):
                return False
            if np.any(np.linalg.eigvalsh(cov) < 0):
                return False
            return True
        
        mesh = o3d.geometry.TriangleMesh()
        for box in boxes:
            # eigenvalue decomposition
            # regularize covariance matrix
            mean = box.center
            cov = box.center_cov
            # continue if cov is <= reg

            if not is_valid_cov(cov, reg=reg):
                raise ValueError("Invalid covariance matrix")
            if np.allclose(cov, 0, atol=reg):
                continue

            cov = cov + np.eye(3) * reg
            eig_val, eig_vec = np.linalg.eigh(cov)

            # eigenvalue must be greater than axis_min
            if axis_min > 0.0:
                eig_val[eig_val < axis_min] = axis_min / n_std

            # unit sphere
            cov_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
            vertices = np.asarray(cov_mesh.vertices)

            # deform sphere
            transform = eig_vec @ np.diag(np.sqrt(eig_val) * n_std)
            vertices = vertices @ transform
            cov_mesh.vertices = o3d.utility.Vector3dVector(vertices)

            # translate sphere
            cov_mesh.translate(mean)
            cov_mesh.rotate(eig_vec, center=mean)

            if lut is not None and box.label_class in lut.labels:
                label = lut.labels[box.label_class]
                c = (label.color[0], label.color[1], label.color[2])
            else:
                if box.confidence == -1.0:
                    c = (0., 1.0, 0.)  # GT: Green
                elif box.confidence >= 0 and box.confidence <= 1.0:
                    c = (1.0, 0., 0.)  # Prediction: red
                else:
                    c = (0.5, 0.5, 0.5)  # Grey

                if hasattr(box, 'is_fusion') and box.is_fusion:
                    c = (1.0, 0.5, 0.0) # orange

            cov_mesh.paint_uniform_color(c)
            mesh += cov_mesh
        mesh.compute_vertex_normals()
        return mesh
    
    @staticmethod
    def plot_bbox_on_img(boxes, img, cam_name, lut=None, thickness=3):
        # use PIL to draw the bounding boxes
        img_pil = Image.fromarray(img) # rgb
        draw = ImageDraw.Draw(img_pil)

        img_pil = Image.fromarray(img) # rgb
        draw = ImageDraw.Draw(img_pil)

        # sort boxes by descending distance to camera, then descending confidence
        for box in boxes:
            if not hasattr(box, 'dis_to_cam'):
                box.dis_to_cam = np.linalg.norm(box.center) # TODO: check how dis_to_cam is calculated in open3d

        boxes = sorted(boxes, key=lambda x: (x.dis_to_cam, x.confidence), reverse=True)
        
        for i, box in enumerate(boxes):
            if not hasattr(box, 'box2d') or not hasattr(box, 'cam_name'):
                continue
            if box.cam_name != cam_name:
                continue
            bbox = box.box2d
            
            if lut is not None and box.label_class in lut.labels:
                label = lut.labels[box.label_class]
                c = (label.color[0], label.color[1], label.color[2])
            else:
                if box.confidence == -1.0:
                    c = (0., 1.0, 0.)  # GT: Green
                elif box.confidence >= 0 and box.confidence <= 1.0:
                    c = (1.0, 0., 0.)  # Prediction: red
                else:
                    c = (0.5, 0.5, 0.5)  # Grey
                    
                if hasattr(box, 'is_fusion') and box.is_fusion:
                    c = (1.0, 0.5, 0.0) # orange

            c = tuple([int(255 * x) for x in c])
            draw.rectangle(bbox, outline=c, width=thickness)
            if box.show_meta and box.confidence >= 0 and box.confidence <= 1.0:
                # select font size based on the size of the image
                font_path = font_manager.findfont(font_manager.FontProperties(family='Arial', weight='bold'), fontext='ttf')
                font_size = min(img.shape[0], img.shape[1]) // 30
                font = ImageFont.truetype(font_path, font_size)
                x = bbox[0]
                y = bbox[1] - font_size
                font_bbox = list(draw.textbbox((x, y), str(box.meta), font=font, align="left"))
                font_bbox[1] -= thickness
                draw.rectangle(font_bbox, fill=c, outline=c, width=thickness)
                draw.text((x, y), str(box.meta), font=font, align="left", fill=(255, 255, 255))
            if box.show_meta and box.confidence == -1.0:
                # plot GT boxes in bottom right corner
                font_path = font_manager.findfont(font_manager.FontProperties(family='Arial', weight='bold'), fontext='ttf')
                font_size = min(img.shape[0], img.shape[1]) // 30
                font = ImageFont.truetype(font_path, font_size)
                x = bbox[2]
                y = bbox[3] + font_size
                font_bbox = list(draw.textbbox((x, y), str(box.meta), font=font, anchor="rb", align="right"))
                font_bbox[3] += thickness
                draw.rectangle(font_bbox, fill=c, outline=c, width=thickness)
                draw.text((x, y), str(box.meta), font=font, anchor="rb", align="right", fill=(255, 255, 255))

        return np.array(img_pil).astype(np.uint8)
    
    @staticmethod
    def enable_meta(boxes, attrs=[], format=[]):
        """Enables meta information for the boxes."""
        
        if not attrs:
            return

        if len(attrs) != len(format):
            raise ValueError("Length of attrs and formats should be the same.")
        
        for box in boxes:
            box.show_meta = True
            box.meta = []
            
            for attr, fmt in zip(attrs, format):
                if hasattr(box, attr):
                    value = getattr(box, attr)
                    formatted_value = fmt.format(value)
                    box.meta.append(formatted_value)
                else:
                    raise ValueError("Box does not have attribute " + attr)
            
            box.meta = ", ".join(box.meta)

    @staticmethod
    def project_to_img(boxes, img, lidar2img_rt=np.ones(4), lut=None, outline_only=False, thickness=3):
        """Returns image with projected 3D bboxes

        Args:
            boxes: the list of bounding boxes
            img: an RGB image
            lidar2img_rt: 4x4 transformation from lidar frame to image plane
            lut: a ml3d.vis.LabelLUT that is used to look up the color based on
                the label_class argument of the BoundingBox3D constructor. If
                not provided, a color of 50% grey will be used. (optional)
        """
        if outline_only:
            prev_arrow_length = []
            for box in boxes: # avoid making a copy
                prev_arrow_length.append(box.arrow_length)
                box.arrow_length = 0.0
            lines = BoundingBox3D.create_lines(boxes, lut, out_format="dict")
            for box in boxes:
                box.arrow_length = prev_arrow_length.pop(0)
        else:
            lines = BoundingBox3D.create_lines(boxes, lut, out_format="dict")
        
        points = lines["vertex_positions"]
        indices = lines["line_indices"]
        colors = lines["line_colors"]

        pts_4d = np.concatenate(
            [points.reshape(-1, 3),
             np.ones((len(boxes) * 14, 1))], axis=-1)
        pts_2d = pts_4d @ lidar2img_rt.T

        pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        imgfov_pts_2d = pts_2d[..., :2].reshape(len(boxes), 14, 2)
        indices_2d = indices[..., :2].reshape(len(boxes), 17, 2)
        colors_2d = colors[..., :3].reshape(len(boxes), 17, 3)

        if outline_only:
            return BoundingBox3D.plot_rect2d_on_img(img,
                                                    len(boxes),
                                                    imgfov_pts_2d,
                                                    colors_2d,
                                                    thickness=thickness)
        
        return BoundingBox3D.plot_rect3d_on_img(img,
                                                len(boxes),
                                                imgfov_pts_2d,
                                                indices_2d,
                                                colors_2d,
                                                thickness=thickness)

    @staticmethod
    def plot_rect2d_on_img(img,
                           num_rects,
                           rect_corners,
                           color=None,
                           thickness=1):
        """
        Plot the 2D bounding box of 3D rectangular on 2D images.

        Args:
            img (numpy.array): The numpy array of image.
            num_rects (int): Number of 3D rectangulars.
            rect_corners (numpy.array): Coordinates of the corners of 3D
                rectangulars. Should be in the shape of [num_rect, 8, 2] or
                [num_rect, 14, 2] if counting arrows.
            color (tuple[int]): The color to draw bboxes. Default: (1.0, 1.0,
                1.0), i.e. white.
            thickness (int, optional): The thickness of bboxes. Default: 1.
        """
        # TODO: use to_kitti_format() to get 2d bbox
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)

        if color is None:
            color = [1.0, 1.0, 1.0]
        
        for i in range(num_rects):
            corners = rect_corners[i].astype(int)

            # Calculate the 2D bounding box from the 3D rectangle corners
            min_x, min_y = np.min(corners, axis=0)
            max_x, max_y = np.max(corners, axis=0)

            # Ignore boxes outside a certain threshold
            interesting_corners_scale = 3.0
            if (min_x < -interesting_corners_scale * img.shape[1] or 
                max_x > interesting_corners_scale * img.shape[1] or 
                min_y < -interesting_corners_scale * img.shape[0] or 
                max_y > interesting_corners_scale * img.shape[0]):
                continue
            
            c = color[-1,-1]
            c = tuple([int(255 * x) for x in c])

            # Draw the bounding rectangle
            draw.rectangle([(min_x, min_y), (max_x, max_y)], outline=c, width=thickness)


        return np.array(img_pil).astype(np.uint8)

    @staticmethod
    def plot_rect3d_on_img(img,
                           num_rects,
                           rect_corners,
                           line_indices,
                           color=None,
                           thickness=1):
        """Plot the boundary lines of 3D rectangular on 2D images.

        Args:
            img (numpy.array): The numpy array of image.
            num_rects (int): Number of 3D rectangulars.
            rect_corners (numpy.array): Coordinates of the corners of 3D
                rectangulars. Should be in the shape of [num_rect, 8, 2] or
                [num_rect, 14, 2] if counting arrows.
            line_indices (numpy.array): indicates connectivity of lines between
                rect_corners.  Should be in the shape of [num_rect, 12, 2] or
                [num_rect, 17, 2] if counting arrows.
            color (tuple[int]): The color to draw bboxes. Default: (1.0, 1.0,
                1.0), i.e. white.
            thickness (int, optional): The thickness of bboxes. Default: 1.
        """
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)

        if color is None:
            color = np.ones((line_indices.shape[0], line_indices.shape[1], 3))
        for i in range(num_rects):
            corners = rect_corners[i].astype(int)
            # ignore boxes outside a certain threshold
            interesting_corners_scale = 3.0
            if min(corners[:, 0]
                  ) < -interesting_corners_scale * img.shape[1] or max(
                      corners[:, 0]
                  ) > interesting_corners_scale * img.shape[1] or min(
                      corners[:, 1]
                  ) < -interesting_corners_scale * img.shape[0] or max(
                      corners[:, 1]) > interesting_corners_scale * img.shape[0]:
                continue
            for j, (start, end) in enumerate(line_indices[i]):
                c = tuple(color[i][j] * 255)  # TODO: not working
                c = (int(c[0]), int(c[1]), int(c[2]))
                if i != 0:
                    pt1 = (corners[(start) % (14 * i),
                                   0], corners[(start) % (14 * i), 1])
                    pt2 = (corners[(end) % (14 * i),
                                   0], corners[(end) % (14 * i), 1])
                else:
                    pt1 = (corners[start, 0], corners[start, 1])
                    pt2 = (corners[end, 0], corners[end, 1])
                draw.line([pt1, pt2], fill=c, width=thickness)
        return np.array(img_pil).astype(np.uint8)
