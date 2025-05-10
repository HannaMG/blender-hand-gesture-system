import joblib
import mediapipe as mp
import cv2
import sys
from math import radians
import bpy
from mathutils import Vector
from mathutils import Matrix
import threading
from bpy_extras import view3d_utils
import numpy as np
import scipy.signal

svm_model_path = "C:\\Users\\hanna\\Documents\\GMU\CS682\\Final_Project\\Mediapipe_code\\mediapipe_hog_svm_model.pkl"

r_gesture = None
l_gesture = None

# Right hand variables
r_pan_vector = None
r_rotate_x = None
r_rotate_y = None
r_rotate_z = None
r_global_hand_pos = None
r_pick_bool = False
r_selected_obj = None
r_change_in_size = None
r_rot_quat = None

# Left hand variables
l_pan_vector = None
l_rotate_x = None
l_rotate_y = None
l_rotate_z = None
l_global_hand_pos = None
l_pick_bool = False
l_selected_obj = None
l_change_in_size = None
l_rot_quat = None

class VIEWPORT_OT_hand_navigate(bpy.types.Operator):
    bl_idname = "view3d.hand_navigate"
    bl_label = "Navigate Viewport"

    _timer = None

    def modal(self, context, event):
        global r_pan_vector
        global r_rotate_x
        global r_rotate_y
        global r_rotate_z
        global r_global_hand_pos
        global r_pick_bool
        global r_selected_obj
        global r_change_in_size
        global r_rot_quat

        global l_pan_vector
        global l_rotate_x
        global l_rotate_y
        global l_rotate_z
        global l_global_hand_pos
        global l_pick_bool
        global l_selected_obj
        global l_change_in_size
        global l_rot_quat

        if event.type == 'TIMER':

            # Right hand

            # Panning the viewport
            if r_pan_vector is not None:
                for area in bpy.context.screen.areas:
                    if area.type == 'VIEW_3D':
                        v3d = area.spaces.active
                        rv3d = v3d.region_3d
                        rv3d.view_location += rv3d.view_rotation @ r_pan_vector
                        area.tag_redraw()
                        break
                r_pan_vector = None

            # Rotating the viewport
            if r_rotate_x is not None and r_rotate_y is not None and r_rotate_z is not None:
                for area in bpy.context.screen.areas:
                    if area.type == 'VIEW_3D':
                        v3d = area.spaces.active
                        rv3d = v3d.region_3d

                        # Get relative axes
                        x_axis = rv3d.view_rotation @ Vector((1, 0, 0))
                        y_axis = rv3d.view_rotation @ Vector((0, 1, 0))
                        z_axis = rv3d.view_rotation @ Vector((0, 0, 1))

                        # Create rotation matrices using relative axes
                        rel_rotate_x = Matrix.Rotation(radians(r_rotate_x), 4, x_axis)    
                        rel_rotate_y = Matrix.Rotation(radians(r_rotate_y), 4, y_axis)
                        rel_rotate_z = Matrix.Rotation(radians(r_rotate_z), 4, z_axis)

                        rot_matrix = (rel_rotate_x @ rel_rotate_y @ rel_rotate_z).to_3x3().to_quaternion()
                        rv3d.view_rotation = rot_matrix @ rv3d.view_rotation
                        area.tag_redraw()
                        break
                r_rotate_x = None
                r_rotate_y = None
                r_rotate_z = None

            # Moving the cursor
            if r_global_hand_pos is not None:
                # Have custom cursor follow the hand position
                for area in bpy.context.screen.areas:
                    if area.type == 'VIEW_3D':
                        v3d = area.spaces.active
                        rv3d = v3d.region_3d
                        
                        region = None
                        for r in area.regions:
                            if r.type == 'WINDOW':
                                region = r

                        x_coord = r_global_hand_pos.x * region.width
                        y_coord = region.height - (r_global_hand_pos.y * region.height)


                        cursor_location = view3d_utils.region_2d_to_location_3d(region, rv3d, (x_coord, y_coord), bpy.context.scene.cursor.location)
                        bpy.context.scene.cursor.location = cursor_location
                        area.tag_redraw()
                        break
                r_global_hand_pos = None

            # Moving selected object
            if r_pick_bool:
                # Convert cursor position to 2D coordinates
                cursor_coords_2d = None
                region = None
                rv3d = None
                for area in bpy.context.screen.areas:
                    if area.type == 'VIEW_3D':
                        v3d = area.spaces.active
                        rv3d = v3d.region_3d

                        for r in area.regions:
                            if r.type == 'WINDOW':
                                region = r

                        cursor_coords_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, bpy.context.scene.cursor.location)

                if r_selected_obj is None:
                    if cursor_coords_2d is not None:
                        # Select object under cursor
                        for obj in bpy.context.scene.objects:
                            threshold = 80
                            # Convert object position to 2D coordinates
                            obj_coords_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, obj.location)

                            if obj_coords_2d is not None:
                                if ((cursor_coords_2d[0] - obj_coords_2d[0])**2 + (cursor_coords_2d[1] - obj_coords_2d[1])**2)**0.5 < threshold:
                                    # Select object if it is not currently selected
                                    if l_selected_obj != obj:
                                        r_selected_obj = obj
                                        break

                if r_selected_obj is not None:

                    # Get relative x-axis
                    z_axis = rv3d.view_rotation @ Vector((0, 0, 1))

                    # Keep object's depth the same
                    if cursor_coords_2d is not None:
                        r_selected_obj.location = view3d_utils.region_2d_to_location_3d(region, rv3d, cursor_coords_2d, (r_selected_obj.location @ z_axis) * z_axis)

                    # Change depth location of object using hand size
                    if r_change_in_size is not None:
                        r_selected_obj.location += -r_change_in_size * z_axis
                        r_change_in_size = None

                    # Change object rotation
                    if r_rot_quat is not None:
                        world_rot = rv3d.view_rotation @ r_rot_quat @ rv3d.view_rotation.inverted()
                        r_selected_obj.rotation_mode = 'QUATERNION'
                        r_selected_obj.rotation_quaternion = world_rot @ r_selected_obj.rotation_quaternion

                        r_rot_quat = None

                    r_pick_bool = False


            # Left hand

            # Panning the viewport
            if l_pan_vector is not None:
                for area in bpy.context.screen.areas:
                    if area.type == 'VIEW_3D':
                        v3d = area.spaces.active
                        rv3d = v3d.region_3d
                        rv3d.view_location += rv3d.view_rotation @ l_pan_vector
                        area.tag_redraw()
                        break
                l_pan_vector = None

            # Rotating the viewport
            if l_rotate_x is not None and l_rotate_y is not None and l_rotate_z is not None:
                for area in bpy.context.screen.areas:
                    if area.type == 'VIEW_3D':
                        v3d = area.spaces.active
                        rv3d = v3d.region_3d

                        # Get relative axes
                        x_axis = rv3d.view_rotation @ Vector((1, 0, 0))
                        y_axis = rv3d.view_rotation @ Vector((0, 1, 0))
                        z_axis = rv3d.view_rotation @ Vector((0, 0, 1))

                        # Create rotation matrices using relative axes
                        rel_rotate_x = Matrix.Rotation(radians(l_rotate_x), 4, x_axis)    
                        rel_rotate_y = Matrix.Rotation(radians(l_rotate_y), 4, y_axis)
                        rel_rotate_z = Matrix.Rotation(radians(l_rotate_z), 4, z_axis)

                        rot_matrix = (rel_rotate_x @ rel_rotate_y @ rel_rotate_z).to_3x3().to_quaternion()
                        rv3d.view_rotation = rot_matrix @ rv3d.view_rotation
                        area.tag_redraw()
                        break
                l_rotate_x = None
                l_rotate_y = None
                l_rotate_z = None

            # Empty object will act as hand cursor
            l_cursor = None
            if "LeftHandCursor" not in bpy.data.objects:
                l_cursor = bpy.data.objects.new('empty', None)
                l_cursor.name = "LeftHandCursor"
                l_cursor.empty_display_size = 0.3
                l_cursor.empty_display_type = 'SPHERE'
                l_cursor.show_in_front = True
                bpy.context.scene.collection.objects.link(l_cursor)
            else:
                l_cursor = bpy.data.objects['LeftHandCursor']

            # Moving the cursor
            if l_global_hand_pos is not None:
                # Have custom cursor follow the hand position
                for area in bpy.context.screen.areas:
                    if area.type == 'VIEW_3D':
                        v3d = area.spaces.active
                        rv3d = v3d.region_3d
                        
                        region = None
                        for r in area.regions:
                            if r.type == 'WINDOW':
                                region = r

                        x_coord = l_global_hand_pos.x * region.width
                        y_coord = region.height - (l_global_hand_pos.y * region.height)


                        cursor_location = view3d_utils.region_2d_to_location_3d(region, rv3d, (x_coord, y_coord), l_cursor.location)
                        l_cursor.location = cursor_location
                        area.tag_redraw()
                        break
                l_global_hand_pos = None

            # Moving selected object
            if l_pick_bool:
                # Convert cursor position to 2D coordinates
                cursor_coords_2d = None
                region = None
                rv3d = None
                for area in bpy.context.screen.areas:
                    if area.type == 'VIEW_3D':
                        v3d = area.spaces.active
                        rv3d = v3d.region_3d

                        for r in area.regions:
                            if r.type == 'WINDOW':
                                region = r

                        cursor_coords_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, l_cursor.location)

                if l_selected_obj is None:
                    if cursor_coords_2d is not None:
                        # Select object under cursor
                        for obj in bpy.context.scene.objects:
                            threshold = 80
                            # Convert object position to 2D coordinates
                            obj_coords_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, obj.location)

                            if obj_coords_2d is not None:
                                if ((cursor_coords_2d[0] - obj_coords_2d[0])**2 + (cursor_coords_2d[1] - obj_coords_2d[1])**2)**0.5 < threshold:
                                    # Select object if it is not currently selected and isn't the cursor
                                    if r_selected_obj != obj and l_cursor != obj:
                                        l_selected_obj = obj
                                        break

                if l_selected_obj is not None:

                    # Get relative x-axis
                    z_axis = rv3d.view_rotation @ Vector((0, 0, 1))

                    # Keep object's depth the same
                    if cursor_coords_2d is not None:
                        l_selected_obj.location = view3d_utils.region_2d_to_location_3d(region, rv3d, cursor_coords_2d, (l_selected_obj.location @ z_axis) * z_axis)

                    # Change depth location of object using hand size
                    if l_change_in_size is not None:
                        l_selected_obj.location += -l_change_in_size * z_axis
                        l_change_in_size = None

                    # Change object rotation
                    if l_rot_quat is not None:
                        world_rot = rv3d.view_rotation @ l_rot_quat @ rv3d.view_rotation.inverted()
                        l_selected_obj.rotation_mode = 'QUATERNION'
                        l_selected_obj.rotation_quaternion = world_rot @ l_selected_obj.rotation_quaternion

                        l_rot_quat = None

                    l_pick_bool = False

        return {'PASS_THROUGH'}
    
    def invoke(self, context, event):
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

def register():
    bpy.utils.register_class(VIEWPORT_OT_hand_navigate)
    bpy.ops.view3d.hand_navigate('INVOKE_DEFAULT')

def unregister():
    bpy.utils.unregister_class(VIEWPORT_OT_hand_navigate)


def smooth(new_value, old_values):
    return 0.35 * new_value + 0.35 * old_values[0] + 0.3 * old_values[1] 


def hands():
    global svm_model_path

    global l_gesture
    global r_gesture

    global r_pan_vector
    global r_rotate_x
    global r_rotate_y     
    global r_rotate_z
    global r_global_hand_pos
    global r_pick_bool
    global r_selected_obj
    global r_change_in_size
    global r_rot_quat

    global l_pan_vector
    global l_rotate_x
    global l_rotate_y     
    global l_rotate_z
    global l_global_hand_pos
    global l_pick_bool
    global l_selected_obj
    global l_change_in_size
    global l_rot_quat

    svm_model = joblib.load(svm_model_path)
    print("Model loaded from 'mediapipe_hog_svm_model.pkl'")

    exit_flag = True
    
    # Right hand previous data
    r_prev_hand_pos = []
    r_prev_hand_size = []
    r_prev_hand_rot = []
    r_prev_landmark_data = []
    r_prev_x_vals = []
    r_prev_y_vals = []

    # Left hand previous data
    l_prev_hand_pos = []
    l_prev_hand_size = []
    l_prev_hand_rot = []
    l_prev_landmark_data = []
    l_prev_x_vals = []
    l_prev_y_vals = []

    while exit_flag:
        cap = cv2.VideoCapture(0)
        with mp.solutions.hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                        # Right hand
                        r_gesture = None
                        if handedness.classification[0].label == 'Right':
                            landmarks_data = []

                            for landmark in hand_landmarks.landmark:
                                # Extract x, y, z coordinates of each landmark and append to landmarks_data list
                                landmarks_data.append([landmark.x, landmark.y, landmark.z])

                            # Smooth out landmark data
                            if len(r_prev_landmark_data) == 2:
                                for i in range(len(landmarks_data)):
                                    landmarks_data[i][0] = smooth(landmarks_data[i][0], [r_prev_landmark_data[0][i][0], r_prev_landmark_data[1][i][0]])
                                    landmarks_data[i][1] = smooth(landmarks_data[i][1], [r_prev_landmark_data[0][i][1], r_prev_landmark_data[1][i][1]])
                                    landmarks_data[i][2] = smooth(landmarks_data[i][2], [r_prev_landmark_data[0][i][2], r_prev_landmark_data[1][i][2]])
                            # Get hand bounding box
                            x_vals = []
                            y_vals = []
                            for landmark in hand_landmarks.landmark:
                                # Extract x, y coordinates of each landmark and append to corresponding list
                                x_vals.append(landmark.x)
                                y_vals.append(landmark.y)

                            # Smooth out x, y landmark data
                            if len(r_prev_x_vals) == 2:
                                for i in range(len(x_vals)):
                                    x_vals[i] = smooth(x_vals[i], [r_prev_x_vals[0][i], r_prev_x_vals[1][i]])
                            if len(r_prev_y_vals) == 2:
                                for i in range(len(y_vals)):
                                    y_vals[i] = smooth(y_vals[i], [r_prev_y_vals[0][i], r_prev_y_vals[1][i]])

                            # Get min and max values for bounding box
                            x_min = min(x_vals)
                            y_min = min(y_vals)
                            x_max = max(x_vals)
                            y_max = max(y_vals)
                            
                            # Convert to pixel coordinates using 
                            height, width, _ = image.shape
                            x_min_pixel = int(x_min * width)
                            x_max_pixel = int(x_max * width)
                            y_min_pixel = int(y_min * height)
                            y_max_pixel = int(y_max * height)
                            
                            # Crop hand region using bounding box
                            hand_img = image[y_min_pixel:y_max_pixel, x_min_pixel:x_max_pixel]
                            # In case of invalid hand image...
                            if hand_img.size == 0:
                                continue
                            
                            # Resize hand image for consistency and convert to grayscale
                            hand_img = cv2.resize(hand_img, (256, 256))
                            hand_img = hand_img[:, :, 0]
                            hand_img = hand_img.astype(np.float32)

                            # In case of invalid size...
                            if hand_img.shape[0] != 256 or hand_img.shape[1] != 256:
                                continue
                            
                            # Compute gradients
                            ix = np.matmul(np.array([[1], [2], [1]]), np.array([[1, 0, -1]]))
                            iy = np.matmul(np.array([[1], [0], [-1]]), np.array([[1, 2, 1]]))
                            deriv_x = scipy.signal.convolve(hand_img, ix, mode='same')
                            deriv_y = scipy.signal.convolve(hand_img, iy, mode='same')
                            grad_mag = np.sqrt(deriv_x**2 + deriv_y**2)
                            grad_ang = np.arctan2(deriv_y, deriv_x)

                            # Convert angles to degrees between 0 to 180
                            grad_ang_deg = np.degrees(grad_ang) % 180

                            # HoG implementation

                            # HoG variables
                            cell_size = 8
                            num_bins = 9
                            bin_size = 20
                            all_histograms = []

                            # Iterate through all 8x8 cells
                            for i in range(32):
                                for j in range(32):
                                    histogram = [0, 0, 0, 0, 0, 0, 0, 0, 0]

                                    # Iterate through single cell
                                    for r in range(i*cell_size, i*cell_size+cell_size):
                                        for c in range(j*cell_size, j*cell_size+cell_size):
                                            mag = grad_mag[r, c]
                                            ang = grad_ang_deg[r, c]

                                            # Get closest bin index and bin
                                            bin_i = round(ang / bin_size)
                                            if bin_i == 9:
                                                bin_i = 0
                                                ang -= 180
                                            bin_at_i = bin_i * bin_size

                                            # Use interpolation
                                            percent_in_neighbor_bin = abs((ang - bin_at_i) / bin_size)
                                            percent_in_bin = 1-percent_in_neighbor_bin

                                            histogram[bin_i] += mag*percent_in_bin
                                            if ang - bin_at_i < 0:
                                                if bin_i - 1 >= 0:
                                                    histogram[bin_i-1] += mag*percent_in_neighbor_bin
                                                else: # Loop around bins
                                                    histogram[num_bins-1] += mag*percent_in_neighbor_bin
                                            elif ang - bin_at_i > 0:
                                                if bin_i + 1 < num_bins:
                                                    histogram[bin_i+1] += mag*percent_in_neighbor_bin
                                                else: # Loop around bins
                                                    histogram[0] += mag*percent_in_neighbor_bin

                                    all_histograms.append(histogram)

                            # Set center hand position from first landmark and smooth it
                            hand_pos = landmarks_data[0]
                            if len(r_prev_hand_pos) == 2:
                                hand_pos[0] = smooth(hand_pos[0], [r_prev_hand_pos[0][0], r_prev_hand_pos[1][0]])
                                hand_pos[1] = smooth(hand_pos[1], [r_prev_hand_pos[0][1], r_prev_hand_pos[1][1]])
                                hand_pos[2] = smooth(hand_pos[2], [r_prev_hand_pos[0][2], r_prev_hand_pos[1][2]])
                            r_global_hand_pos = Vector((hand_pos[0], hand_pos[1], hand_pos[2]))

                            # Set hand size based on bounding box diagonal and smooth it
                            hand_size = ((x_max - x_min)**2 + (y_max - y_min)**2)**0.5
                            if len(r_prev_hand_size) == 2:
                                hand_size = smooth(hand_size, r_prev_hand_size)

                            # Set hand rotation by calculating relative hand axes and smooth it
                            wrist_3d_coords = Vector((landmarks_data[0][0], landmarks_data[0][1], landmarks_data[0][2])) 
                            pinky_3d_coords = Vector((landmarks_data[17][0], landmarks_data[17][1], landmarks_data[17][2]))
                            index_3d_coords = Vector((landmarks_data[5][0], landmarks_data[5][1], landmarks_data[5][2])) 
                            hand_vector_1 = (pinky_3d_coords - wrist_3d_coords).normalized()
                            hand_vector_2 = (index_3d_coords - wrist_3d_coords).normalized()
                            palm_normal = hand_vector_1.cross(hand_vector_2).normalized()
                            hand_axis_3 = palm_normal.cross(hand_vector_1).normalized()
                            hand_rot = [-hand_vector_1, palm_normal, hand_axis_3]
                            if len(r_prev_hand_rot) == 2:
                                hand_rot[0] = smooth(hand_rot[0], [r_prev_hand_rot[0][0], r_prev_hand_rot[1][0]])
                                hand_rot[1] = smooth(hand_rot[1], [r_prev_hand_rot[0][1], r_prev_hand_rot[1][1]])
                                hand_rot[2] = smooth(hand_rot[2], [r_prev_hand_rot[0][2], r_prev_hand_rot[1][2]])

                            # Get gesture prediction and draw hand landmarks
                            r_gesture = svm_model.predict([[i for s in all_histograms for i in s]])[0]
                            print(f"Predicted gesture: {r_gesture}")
                            mp.solutions.drawing_utils.draw_landmarks(
                                image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)


                            # Blender commands based on hand prediction
                            if r_gesture == 'open_hand':
                                cv2.putText(image, 'Open Hand', (int(image.shape[1] * 0.65), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                                # Release any picked up objects
                                r_selected_obj = None

                                # Base hand gesture, nothing else happens

                            elif r_gesture == 'fist':
                                cv2.putText(image, 'Fist', (int(image.shape[1] * 0.65), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                                # Can't pan with both hands
                                if l_gesture != 'fist':

                                    # Pan viewport based on fist gesture
                                    if r_prev_hand_pos and r_prev_hand_size:
                                        dx = (hand_pos[0] - r_prev_hand_pos[0][0]) * 100
                                        dy = (hand_pos[1] - r_prev_hand_pos[0][1]) * 100
                                        dz = (hand_size - r_prev_hand_size[0]) * 300

                                        r_pan_vector = Vector((-dx, dy, dz))

                                    # Release any picked up objects
                                    r_selected_obj = None

                            elif r_gesture == 'pinch':
                                cv2.putText(image, 'Pinch', (int(image.shape[1] * 0.65), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                                # Can't rotate with both hands
                                if l_gesture != 'pinch':

                                    # Rotate viewport based on pinch gesture
                                    if r_prev_hand_pos and r_prev_hand_size:
                                        dx = (hand_pos[0] - r_prev_hand_pos[0][0]) * 200
                                        dy = (hand_pos[1] - r_prev_hand_pos[0][1]) * 200
                                        dz = (hand_size - r_prev_hand_size[0]) * 1000

                                        r_rotate_x = -dy
                                        r_rotate_y = -dx
                                        r_rotate_z = dz

                                    # Release any picked up objects
                                    r_selected_obj = None

                            elif r_gesture == 'pick':
                                cv2.putText(image, 'Pick', (int(image.shape[1] * 0.65), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                                
                                if r_prev_hand_size and r_prev_hand_rot:
                                    r_change_in_size = (hand_size - r_prev_hand_size[0]) * 300
                                    
                                    rot = Matrix((hand_rot[0], hand_rot[1], hand_rot[2])).to_quaternion()
                                    prev_rot = Matrix((r_prev_hand_rot[0][0], r_prev_hand_rot[0][1], r_prev_hand_rot[0][2])).to_quaternion()

                                    r_rot_quat = rot @ prev_rot.inverted()

                                    r_pick_bool = True

                            # Save at most two hand positions
                            r_prev_hand_pos.insert(0, hand_pos)
                            if len(r_prev_hand_pos) > 2:
                                r_prev_hand_pos.pop()
                            
                            # Save at most two hand sizes
                            r_prev_hand_size.insert(0, hand_size)
                            if len(r_prev_hand_size) > 2:
                                r_prev_hand_size.pop()

                            # Save at most two hand rotations
                            r_prev_hand_rot.insert(0, hand_rot)
                            if len(r_prev_hand_rot) > 2:
                                r_prev_hand_rot.pop()

                            # Save at most two sets of hand landmarks
                            r_prev_landmark_data.insert(0, landmarks_data)
                            if len(r_prev_landmark_data) > 2:
                                r_prev_landmark_data.pop()

                            # Save at most two sets of x values
                            r_prev_x_vals.insert(0, x_vals)          
                            if len(r_prev_x_vals) > 2:
                                r_prev_x_vals.pop()
                            
                            # Save at most two sets of x values
                            r_prev_y_vals.insert(0, y_vals)          
                            if len(r_prev_y_vals) > 2:
                                r_prev_y_vals.pop()

                        # Left hand
                        l_gesture = None
                        if handedness.classification[0].label == 'Left':
                            landmarks_data = []

                            for landmark in hand_landmarks.landmark:
                                # Extract x, y, z coordinates of each landmark and append to landmarks_data list
                                landmarks_data.append([landmark.x, landmark.y, landmark.z])

                            # Smooth out landmark data
                            if len(l_prev_landmark_data) == 2:
                                for i in range(len(landmarks_data)):
                                    landmarks_data[i][0] = smooth(landmarks_data[i][0], [l_prev_landmark_data[0][i][0], l_prev_landmark_data[1][i][0]])
                                    landmarks_data[i][1] = smooth(landmarks_data[i][1], [l_prev_landmark_data[0][i][1], l_prev_landmark_data[1][i][1]])
                                    landmarks_data[i][2] = smooth(landmarks_data[i][2], [l_prev_landmark_data[0][i][2], l_prev_landmark_data[1][i][2]])
                            # Get hand bounding box
                            x_vals = []
                            y_vals = []
                            for landmark in hand_landmarks.landmark:
                                # Extract x, y coordinates of each landmark and append to corresponding list
                                x_vals.append(landmark.x)
                                y_vals.append(landmark.y)

                            # Smooth out x, y landmark data
                            if len(l_prev_x_vals) == 2:
                                for i in range(len(x_vals)):
                                    x_vals[i] = smooth(x_vals[i], [l_prev_x_vals[0][i], l_prev_x_vals[1][i]])
                            if len(l_prev_y_vals) == 2:
                                for i in range(len(y_vals)):
                                    y_vals[i] = smooth(y_vals[i], [l_prev_y_vals[0][i], l_prev_y_vals[1][i]])

                            # Get min and max values for bounding box
                            x_min = min(x_vals)
                            y_min = min(y_vals)
                            x_max = max(x_vals)
                            y_max = max(y_vals)
                            
                            # Convert to pixel coordinates using 
                            height, width, _ = image.shape
                            x_min_pixel = int(x_min * width)
                            x_max_pixel = int(x_max * width)
                            y_min_pixel = int(y_min * height)
                            y_max_pixel = int(y_max * height)
                            
                            # Crop hand region using bounding box
                            hand_img = image[y_min_pixel:y_max_pixel, x_min_pixel:x_max_pixel]
                            # In case of invalid hand image...
                            if hand_img.size == 0:
                                continue
                            
                            # Resize hand image for consistency and convert to grayscale
                            hand_img = cv2.resize(hand_img, (256, 256))
                            hand_img = hand_img[:, :, 0]
                            hand_img = hand_img.astype(np.float32)

                            # In case of invalid size...
                            if hand_img.shape[0] != 256 or hand_img.shape[1] != 256:
                                continue
                            
                            # Compute gradients
                            ix = np.matmul(np.array([[1], [2], [1]]), np.array([[1, 0, -1]]))
                            iy = np.matmul(np.array([[1], [0], [-1]]), np.array([[1, 2, 1]]))
                            deriv_x = scipy.signal.convolve(hand_img, ix, mode='same')
                            deriv_y = scipy.signal.convolve(hand_img, iy, mode='same')
                            grad_mag = np.sqrt(deriv_x**2 + deriv_y**2)
                            grad_ang = np.arctan2(deriv_y, deriv_x)

                            # Convert angles to degrees between 0 to 180
                            grad_ang_deg = np.degrees(grad_ang) % 180

                            # HoG implementation

                            # HoG variables
                            cell_size = 8
                            num_bins = 9
                            bin_size = 20
                            all_histograms = []

                            # Iterate through all 8x8 cells
                            for i in range(32):
                                for j in range(32):
                                    histogram = [0, 0, 0, 0, 0, 0, 0, 0, 0]

                                    # Iterate through single cell
                                    for r in range(i*cell_size, i*cell_size+cell_size):
                                        for c in range(j*cell_size, j*cell_size+cell_size):
                                            mag = grad_mag[r, c]
                                            ang = grad_ang_deg[r, c]

                                            # Get closest bin index and bin
                                            bin_i = round(ang / bin_size)
                                            if bin_i == 9:
                                                bin_i = 0
                                                ang -= 180
                                            bin_at_i = bin_i * bin_size

                                            # Use interpolation
                                            percent_in_neighbor_bin = abs((ang - bin_at_i) / bin_size)
                                            percent_in_bin = 1-percent_in_neighbor_bin

                                            histogram[bin_i] += mag*percent_in_bin
                                            if ang - bin_at_i < 0:
                                                if bin_i - 1 >= 0:
                                                    histogram[bin_i-1] += mag*percent_in_neighbor_bin
                                                else: # Loop around bins
                                                    histogram[num_bins-1] += mag*percent_in_neighbor_bin
                                            elif ang - bin_at_i > 0:
                                                if bin_i + 1 < num_bins:
                                                    histogram[bin_i+1] += mag*percent_in_neighbor_bin
                                                else: # Loop around bins
                                                    histogram[0] += mag*percent_in_neighbor_bin

                                    all_histograms.append(histogram)

                            # Set center hand position from first landmark and smooth it
                            hand_pos = landmarks_data[0]
                            if len(l_prev_hand_pos) == 2:
                                hand_pos[0] = smooth(hand_pos[0], [l_prev_hand_pos[0][0], l_prev_hand_pos[1][0]])
                                hand_pos[1] = smooth(hand_pos[1], [l_prev_hand_pos[0][1], l_prev_hand_pos[1][1]])
                                hand_pos[2] = smooth(hand_pos[2], [l_prev_hand_pos[0][2], l_prev_hand_pos[1][2]])
                            l_global_hand_pos = Vector((hand_pos[0], hand_pos[1], hand_pos[2]))

                            # Set hand size based on bounding box diagonal and smooth it
                            hand_size = ((x_max - x_min)**2 + (y_max - y_min)**2)**0.5
                            if len(l_prev_hand_size) == 2:
                                hand_size = smooth(hand_size, l_prev_hand_size)

                            # Set hand rotation by calculating relative hand axes and smooth it
                            wrist_3d_coords = Vector((landmarks_data[0][0], landmarks_data[0][1], landmarks_data[0][2])) 
                            pinky_3d_coords = Vector((landmarks_data[17][0], landmarks_data[17][1], landmarks_data[17][2]))
                            index_3d_coords = Vector((landmarks_data[5][0], landmarks_data[5][1], landmarks_data[5][2])) 
                            hand_vector_1 = (pinky_3d_coords - wrist_3d_coords).normalized()
                            hand_vector_2 = (index_3d_coords - wrist_3d_coords).normalized()
                            palm_normal = hand_vector_1.cross(hand_vector_2).normalized()
                            hand_axis_3 = palm_normal.cross(hand_vector_1).normalized()
                            hand_rot = [-hand_vector_1, palm_normal, hand_axis_3]
                            if len(l_prev_hand_rot) == 2:
                                hand_rot[0] = smooth(hand_rot[0], [l_prev_hand_rot[0][0], l_prev_hand_rot[1][0]])
                                hand_rot[1] = smooth(hand_rot[1], [l_prev_hand_rot[0][1], l_prev_hand_rot[1][1]])
                                hand_rot[2] = smooth(hand_rot[2], [l_prev_hand_rot[0][2], l_prev_hand_rot[1][2]])

                            # Get gesture prediction and draw hand landmarks
                            l_gesture = svm_model.predict([[i for s in all_histograms for i in s]])[0]
                            print(f"Predicted gesture: {l_gesture}")
                            mp.solutions.drawing_utils.draw_landmarks(
                                image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)


                            # Blender commands based on hand prediction
                            if l_gesture == 'open_hand':
                                cv2.putText(image, 'Open Hand', (int(image.shape[1] * 0.05), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                                # Release any picked up objects
                                l_selected_obj = None

                                # Base hand gesture, nothing else happens

                            elif l_gesture == 'fist':
                                cv2.putText(image, 'Fist', (int(image.shape[1] * 0.05), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                                # Can't pan with both hands
                                if r_gesture != 'fist':

                                    # Pan viewport based on fist gesture
                                    if l_prev_hand_pos and l_prev_hand_size:
                                        dx = (hand_pos[0] - l_prev_hand_pos[0][0]) * 100
                                        dy = (hand_pos[1] - l_prev_hand_pos[0][1]) * 100
                                        dz = (hand_size - l_prev_hand_size[0]) * 300

                                        l_pan_vector = Vector((-dx, dy, dz))

                                    # Release any picked up objects
                                    l_selected_obj = None

                            elif l_gesture == 'pinch':
                                cv2.putText(image, 'Pinch', (int(image.shape[1] * 0.05), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                                # Can't rotate with both hands
                                if r_gesture != 'pinch':

                                    # Rotate viewport based on pinch gesture
                                    if l_prev_hand_pos and l_prev_hand_size:
                                        dx = (hand_pos[0] - l_prev_hand_pos[0][0]) * 200
                                        dy = (hand_pos[1] - l_prev_hand_pos[0][1]) * 200
                                        dz = (hand_size - l_prev_hand_size[0]) * 1000

                                        l_rotate_x = -dy
                                        l_rotate_y = -dx
                                        l_rotate_z = dz

                                    # Release any picked up objects
                                    l_selected_obj = None

                            elif l_gesture == 'pick':
                                cv2.putText(image, 'Pick', (int(image.shape[1] * 0.05), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                                
                                if l_prev_hand_size and l_prev_hand_rot:
                                    l_change_in_size = (hand_size - l_prev_hand_size[0]) * 300
                                    
                                    rot = Matrix((hand_rot[0], hand_rot[1], hand_rot[2])).to_quaternion()
                                    prev_rot = Matrix((l_prev_hand_rot[0][0], l_prev_hand_rot[0][1], l_prev_hand_rot[0][2])).to_quaternion()

                                    l_rot_quat = rot @ prev_rot.inverted()

                                    l_pick_bool = True

                            # Save at most two hand positions
                            l_prev_hand_pos.insert(0, hand_pos)
                            if len(l_prev_hand_pos) > 2:
                                l_prev_hand_pos.pop()
                            
                            # Save at most two hand sizes
                            l_prev_hand_size.insert(0, hand_size)
                            if len(l_prev_hand_size) > 2:
                                l_prev_hand_size.pop()

                            # Save at most two hand rotations
                            l_prev_hand_rot.insert(0, hand_rot)
                            if len(l_prev_hand_rot) > 2:
                                l_prev_hand_rot.pop()

                            # Save at most two sets of hand landmarks
                            l_prev_landmark_data.insert(0, landmarks_data)
                            if len(l_prev_landmark_data) > 2:
                                l_prev_landmark_data.pop()

                cv2.imshow('MediaPipe Hands', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    print("ESC key pressed. Exiting...")
                    cap.release()
                    cv2.destroyAllWindows()

                    # Unregister the blender operators
                    unregister()

                    exit_flag = False


def main():
    register()

    t = threading.Thread(target=hands, daemon=True)
    t.start()
  

if __name__ == "__main__":
    main()