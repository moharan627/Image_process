class kmeans:
    def __init__(self,path):
        self.img = cv2.imread(path)
        self.h,self.w = self.img.shape[:2]
        self.Colors = self.img.reshape(-1,3).astype(np.float32)
        return
    def imgshow(self,img, Name="Image"):
        cv2.namedWindow(Name, cv2.WINDOW_NORMAL)
        cv2.imshow(Name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    def run(self,K=5,num=10,change=1.0):
        param = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, num, change)
        _,labels,centers = cv2.kmeans(
            self.Colors, K, None, param, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS
        )
        labels = labels.squeeze(axis=1)
        centers = centers.astype(np.uint8)
        output = centers[labels].reshape(self.img.shape)
        self.imgshow(output)
        return
