import numpy as np
from scipy.spatial import distance_matrix


def clean_data(x,y,k=6,smote_first=True):
    
    n_inp, n_feat = x.shape
    
    if smote_first==True:
        x_new,y_new = SMOTE(x,y)
        x_new,y_new = remove_tomek(x_new,y_new)
    else:
        x_new,y_new = remove_tomek(x,y)
        x_new,y_new = SMOTE(x_new,y_new,k=k)
    
    return x_new,y_new



def remove_tomek(x,y,k=6):
    
    n_inp,n_feat = x.shape
    dist = distance_matrix(x,x)
    remove = []
    cnt = 0
    for ex in range(n_inp):
        if(y[ex]==1):
            cnt = cnt + 1
            nn = np.argsort(dist[ex,:])[1:k]
            for i in range(k-1):
                if(y[nn[i]]==0):
                    remove.append(nn[i])
                else:
                    break
    print("removing %d samples"%(len(remove)))
    uni_remove = []
    for ele in remove:
        if ele not in uni_remove:
            uni_remove.append(ele)
    remove = np.asarray(uni_remove)

    print remove
    xnew = np.delete(x,remove,0)
    ynew = np.delete(y,remove,0)
    return xnew,ynew


# In[8]:


def SMOTE(x,y,k=6,h=10,alpha = 0.5):
        
    ind_interpolate = []
    no_interpolate = []
    new_samples = []
    dist = distance_matrix(x,x)
    n_inp,n_feat = x.shape
    for ex in range(n_inp):
        if(y[ex]==1):
            k_indices = np.argsort(dist[ex,:])[1:k+1] #because one will be itself, so 0
            k_labels = y[k_indices]
            no_min_neighbours = sum(k_labels)
            if(sum(no_min_neighbours < int(k/2))):
                ind_interpolate.append(ex)
                no_interpolate.append(int(k/2) - no_min_neighbours)
     
    # print(ind_interpolate,no_interpolate)
    ind_interpolate = np.asarray(ind_interpolate).astype(int)
   
    for ex in range((ind_interpolate.shape[0])):
        ind_tmp = ind_interpolate[ex]
        ktmp = int(no_interpolate[ex][0])
        
        get_nearest = np.argsort(dist[ind_tmp,:])[2:h+2]

        np.random.shuffle(get_nearest)
        get_nearest = get_nearest[:ktmp]
        for it in range(ktmp):
            tmp_vec = x[ind_tmp] + alpha*(x[get_nearest[it]] - x[ind_tmp]) 
            new_samples.append(tmp_vec)
    print("Adding %d new samples"%(len(new_samples)))
    new_samples = np.asarray(new_samples)
    
    xnew = np.concatenate((x,new_samples),0)
    ynew = np.concatenate((y,np.ones((new_samples.shape[0],1))),0)
    return xnew,ynew
