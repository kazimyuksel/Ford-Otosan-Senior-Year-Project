# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:45:22 2018

@author: Kazım
"""
import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np

#DIM=(640, 480)
DIM = (640,480)
#K=np.array([[434.13357687078997, 0.0, 330.3849334274633], [0.0, 432.85515403391497, 217.25232115248036], [0.0, 0.0, 1.0]])
#D=np.array([[-0.05799800494225399], [-0.06928672608407957], [-0.14196839865300295], [0.105791024909325]])
#DIM=(640, 480)
#K=np.array([[446.0415983439573, 0.0, 345.22721101897287], [0.0, 443.5367135649637, 234.08160715636996], [0.0, 0.0, 1.0]])
#D=np.array([[-0.054112643073989664], [-0.17710408468773384], [0.07055299685877424], [-0.021227611300060692]])

#K=np.array([[434.69582589661127, 0.0, 329.88593484416015], [0.0, 432.9282346935097, 218.16156689618913], [0.0, 0.0, 1.0]])
#D=np.array([[-0.08421582695848977], [0.1072058128379107], [-0.5622664731446513], [0.42955908638253876]])


#K=np.array([[438.01927267751597, 0.0, 347.0657638797564], [0.0, 434.6231652000436, 237.95133382361726], [0.0, 0.0, 1.0]])
#D=np.array([[-0.0724412329342678], [-0.06048517247364126], [-0.3101397588400879], [0.3939775996094876]])

#K=np.array([[435.68176391602776, 0.0, 347.4413654033962], [0.0, 433.7588875103419, 237.59207632751657], [0.0, 0.0, 1.0]])
#D=np.array([[-0.09710360343338054], [0.03841299365263533], [-0.41640088900412864], [0.4726468600969319]])


#good
K=np.array([[439.5145243002018, 0.0, 347.18970959146174], [0.0, 437.1921587283635, 234.42807641535975], [0.0, 0.0, 1.0]])
D=np.array([[-0.08684233441101091], [0.014219264604985508], [-0.2898478465655217], [0.20792055995078929]])

#K=np.array([[434.44785360884276, 0.0, 350.6082294077779], [0.0, 433.10305725001643, 235.8533640785707], [0.0, 0.0, 1.0]])
#K=np.array([[438.6940973222173, 0.0, 352.9882830695308], [0.0, 436.6226025519875, 235.63386427436632], [0.0, 0.0, 1.0]])
#K=np.array([[440, 0.0, 353], [0.0, 440, 235], [0.0, 0.0, 1.0]])

#D=np.array([[-0.06769670688734186], [-0.06327066007533949], [-0.0763622005752789], [0.04269603985442765]])
#D=np.array([[-0.06769670688734186], [-0.06327066007533949], [-0.0763622005752789], [0.04269603985442765]])

#GOOD
K=np.array([[433.1789201825628, 0.0, 336.0326551869997], [0.0, 430.52735997037956, 216.5706380556904], [0.0, 0.0, 1.0]])
D=np.array([[-0.05280848062554133], [-0.13174782579340452], [-0.028567480262252885], [0.060944836127339065]])

#MEH
#K=np.array([[436.46478410158636, 0.0, 333.39322743457586], [0.0, 434.80640948414174, 218.11527189185384], [0.0, 0.0, 1.0]])
#D=np.array([[-0.053189958213718705], [-0.11944470825468673], [-0.20146493508141508], [0.33412875845595397]])

#MEH2
#K=np.array([[431.07098259033967, 0.0, 338.73006296023584], [0.0, 429.7719846375168, 217.91570328927065], [0.0, 0.0, 1.0]])
#D=np.array([[-0.025724710810226488], [-0.26334697390343464], [0.24762247239131405], [-0.1634226540006772]])

#K=np.array([[435.00244935238015, 0.0, 343.7835177901415], [0.0, 433.64060679959596, 235.25535555556823], [0.0, 0.0, 1.0]])
#D=np.array([[-0.04613685654079228], [-0.20326848435418557], [0.16861775401629442], [-0.1048775975697199]])

def undistort(img_array, balance=1.0, dim2=None, dim3=None):
    img = img_array
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img



video_capture = cv2.VideoCapture(0)
counter = 1
file_dir = 'D:\\FordOtosan\\distorsion_calibration_images\\'
while True:
    ret, rgb = video_capture.read()
    rgb = undistort(rgb)
    cv2.imshow('image',rgb)
    if cv2.waitKey(27) & 0xFF == ord('q') :
        break
    if cv2.waitKey(27) & 0xFF == ord('s') :
        path = file_dir + 'image '+str(counter)+'.jpg'
        cv2.imwrite(path,rgb)
        print(counter)
        counter = counter + 1
        
video_capture.release()
cv2.destroyAllWindows()