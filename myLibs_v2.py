import numpy as np 

class genPoints:

    @staticmethod
    def Line(N=100, K=3, D=2):

        P = np.zeros((N*K, D))
        L = np.zeros(N*K, dtype=np.uint8)
        t = np.linspace(-20, 20, N)
        reshapedT = t.reshape((1,N))

        for j in range(K):

            cooef = 2*np.random.rand(D,1)-1
            cooef /= np.linalg.norm(cooef)
            ix = range(N*j, N*(j+1))

            
            #P[ix] = ((cooef@np.vstack([np.ones((1,N)),reshapedT])) + np.random.randn(D,N)).T
            P[ix] = (cooef@reshapedT + (2*np.random.randn(D,N)-1)).T          
            L[ix] = j
        return P, L
    @staticmethod
    
    def monoLine(N=100, D=2):
        K = 1
        P = np.zeros((N*K))
        L = np.zeros(N*K, dtype=np.uint8)
        t = np.linspace(-20, 20, N)
        reshapedT = t.reshape((1,N))

        for j in range(K):

            cooef = 2*np.random.rand(D,1)-1
            cooef /= np.linalg.norm(cooef)
            ix = range(N*j, N*(j+1))

            
            #P[ix] = ((cooef@np.vstack([np.ones((1,N)),reshapedT])) + np.random.randn(D,N)).T
            P[ix] = (cooef@reshapedT + (2*np.random.randn(D,N)-1)).T          
            L[ix] = j
        return P, L
    @staticmethod
    def Circle(N=100, K=3, D=2):

        P = np.zeros((N*K, 2))
        L = np.zeros(N*K, dtype=np.uint8)

        for j in range(K):

            r = 3*(j+1)

            ix = range(N*j, N*(j+1))

            theta = 2*np.pi*np.random.rand(N)

            radius = r + 0.3*np.random.randn(N)

            x = radius*np.cos(theta)
            y = radius*np.sin(theta)

            P[ix] = np.c_[x, y]
            L[ix] = j

        return P, L

    @staticmethod
    def Cloud(N=100, K=3, D=2):

        P = np.zeros((N*K, D))
        L = np.zeros(N*K, dtype=np.uint8)

        for j in range(K):
            theta = 2*np.pi*j/K
            if D == 3: 
                psi = theta*np.pi/2 
            else: psi = np.pi/2
            
            center_x = 2*np.cos(theta)*np.sin(psi)
            center_y = 2*np.sin(theta)*np.sin(psi)
            center_z = 2*np.cos(psi)
            r = np.zeros(N) 
            for i in range(len(r)): 
                r[i] = np.random.randn()*0.4 
            ix = range(N*j, N*(j+1))
            tTheta = np.linspace(0,2*3.1415,N) + np.random.randn(N)*2 # theta 
            tPsi = np.linspace(0,2*3.1415,N) + np.random.randn(N)*2 # theta 
            if D == 3: 
                P[ix] = np.c_[r*np.sin(tTheta)*np.sin(tPsi) + center_x,
                            r*np.cos(tTheta)*np.sin(tPsi) + center_y,
                            r*np.cos(tPsi) + center_z]
            else: 
                P[ix] = np.c_[r*np.sin(tTheta) + center_x,
                            r*np.cos(tTheta) + center_y]
            
            

            L[ix] = j

        return P, L

    @staticmethod
    def Spiral(N=100, K=3, D=2):

        P = np.zeros((N*K, 2))
        L = np.zeros(N*K, dtype=np.uint8)

        for j in range(K):

            ix = range(N*j, N*(j+1))

            r = np.linspace(0, 1, N)

            t = (
                np.linspace(j*4, (j+1)*4, N)
                + np.random.randn(N)*0.2
            )

            P[ix] = np.c_[
                r*np.sin(t),
                r*np.cos(t)
            ]

            L[ix] = j

        return P, L
class kMean:
    @staticmethod
    def myKMean(datas, ite=100, kPredict = 5 ):
        clusters = np.zeros(datas.shape[0])
        randomIndices = np.random.choice(datas.shape[0], size=kPredict, replace=False)
        centroidPoints = datas[randomIndices]
        for turn in range(ite):
            distances = np.linalg.norm(datas[:, None, :] - centroidPoints, axis=2)
            clusters = np.argmin(distances, axis=1, keepdims=True)
            oldCentroids = centroidPoints.copy()
            for k in range(kPredict):
                mask = (clusters == k).squeeze()
                pointQuantity = np.sum(clusters == k)
                if pointQuantity > 0:
                    centroidPoints[k] = np.sum(datas[mask], axis=0) / pointQuantity
                else:
                    centroidPoints[k] = oldCentroids[k]
        return clusters, centroidPoints
    
    @staticmethod
    def silhouetteScore(datas, clusters, centroidPoints):
        if hasattr(clusters, 'squeeze'):
            clusters = clusters.squeeze()
        if np.ndim(clusters) > 1:
            clusters = clusters.ravel()
            
        # Tính khoảng cách từ các điểm tới centroids
        distancePCentroid = np.linalg.norm(datas[:,None,:]-centroidPoints,axis=2)
        # Tìm cụm khác nó nhưng gần nó nhất
        if centroidPoints.shape[0] >= 2:
            closetCluster = np.argpartition(distancePCentroid, kth=1, axis=1)[:, 1][:,None]
            outClusterMask = clusters[None,:] == closetCluster
        else:
            outClusterMask = clusters[None,:] != clusters
        # Tìm cụm chính nó
        inClusterMask = clusters[None,:] == clusters[:, None]
        

        # Tính ma trận khoảng cách giữa các cặp điểm
        distanceMatrix = np.linalg.norm(datas[:, None, :] - datas[None, :, :], axis=2)

        # Các phép tính toán ma trận vuông 2D thực hiện song song
        countIn = np.sum(inClusterMask, axis=1)
        countOut = np.sum(outClusterMask, axis=1)

        sumIn = np.sum(distanceMatrix * inClusterMask, axis=1)
        sumOut = np.sum(distanceMatrix * outClusterMask, axis=1)

        # Tính toán an toàn, triệt tiêu chia cho 0
        a = np.where(countIn > 1, sumIn / (countIn - 1), 0)
        b = np.where(countOut > 0, sumOut / countOut, 0)

        max_ba = np.maximum(b, a)
        silhouetteElements = np.where(max_ba > 0, (b - a) / max_ba, 0)
        
        return np.mean(silhouetteElements)
    def elbowScore(datas,centroidPoints):
        distancePCentroid = np.linalg.norm(datas[:,None,:]-centroidPoints,axis=2)**2
        elbowScore = np.sum(np.min(distancePCentroid,axis=1))
        return elbowScore

    
