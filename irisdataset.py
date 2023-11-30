from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from sklearn import preprocessing
from sklearn import datasets
from sklearn import svm
from dgrid import DGrid
from umap import UMAP
from PIL import Image
import matplotlib.patches as mpatches
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import pandas as pd
import numpy as np
import numpy as np
import matplotlib
import time
import math
import shap

start_time = time.time()

def draw_porcupineglyph(x, y, size, data, axes, true_y, facecolor, alpha, base, i_, max_i):
    nr_points = len(data)
    increments = 360.0 / (nr_points*2)

    Path = mpath.Path
    path_data = [None]*nr_points
    if(i_>1):
      x_ = x + (size / 4) * math.cos(math.radians((increments)))
      y_ = y + (size / 4) * math.sin(math.radians((increments)))
      axes.text(x_, y_, str(i_), color="black", fontsize=6)

    for i in range(nr_points):
        path_data[i] = [(Path.MOVETO, (x, y))]
        x_ = x + (size / 2) * math.cos(math.radians((i*2 * increments)))
        y_ = y + (size / 2) * math.sin(math.radians((i*2 * increments)))
        path_data[i].append((Path.LINETO, (x_, y_)))
        codes, verts = zip(*path_data[i])
        path = mpath.Path(verts, codes)
        if(i==true_y):
            patch = mpatches.PathPatch(path, linewidth=1.2, edgecolor='black', linestyle='solid', alpha=alpha)
        else:
            patch = mpatches.PathPatch(path, linewidth=0.5, edgecolor='black', linestyle='solid', alpha=alpha)
        axes.add_patch(patch)

    path_data = [None]*nr_points

    for i in range(nr_points):

        path_data[i] = [(Path.MOVETO, (x + ((size / 2) * math.cos(math.radians((i*2-1)*increments))) * base,
                                      y + ((size / 2) * math.sin(math.radians((i*2-1)*increments))) * base
                                      ))]
        if(data[i] !=0):
          x_ = x + math.cos(math.radians(2*i * increments)) * ((data[i] * ((size / 2) - math.cos(math.radians(increments))*base)) + math.cos(math.radians(increments))*base)
          y_ = y + math.sin(math.radians(2*i * increments)) * ((data[i] * ((size / 2) - math.cos(math.radians(increments))*base)) + math.cos(math.radians(increments))*base)
          path_data[i].append((Path.LINETO, (x_, y_)))
        x_ = x + ((size / 2) * math.cos(math.radians(((2*i+1) * increments)))) * base
        y_ = y + ((size / 2) * math.sin(math.radians(((2*i+1) * increments)))) * base
        path_data[i].append((Path.LINETO, (x_, y_)))
        path_data[i].append((Path.CLOSEPOLY, (x + ((size / 2) * math.cos(math.radians((i*2+1)*increments))) * base,
                                                 y + ((size / 2) * math.sin(math.radians((i*2+1)*increments))) * base
                                                )))

        codes, verts = zip(*path_data[i])
        path = Path(verts, codes)
        patch = mpatches.PathPatch(path, facecolor=facecolor, linewidth=0.5, edgecolor='black', alpha=alpha)
        axes.add_patch(patch)

    path_c = [(Path.MOVETO, (x + ((size / 2) * math.cos(math.radians((-1)*increments))) * base,
                             y + ((size / 2) * math.sin(math.radians((-1)*increments))) * base
                                      ))]
    for i in range(nr_points):
      x_ = x + ((size / 2) * math.cos(math.radians((i*2+1)*increments))) * base
      y_ = y + ((size / 2) * math.sin(math.radians((i*2+1)*increments))) * base
      path_c.append((Path.LINETO, (x_, y_)))

    path_c.append((Path.CLOSEPOLY, (x + ((size / 2) * math.cos(math.radians((-1)*increments))) * base,
                                y + ((size / 2) * math.sin(math.radians((-1)*increments))) * base
                                      )))
    
    codes, verts = zip(*path_c)
    path = Path(verts, codes)
    if max_i == 1:
       color = 1
    else:
      color = 1-((i_-1)/(max_i-1))
    patch = mpatches.PathPatch(path, facecolor=str(color), linewidth=0.5, edgecolor='black', alpha=1)
    axes.add_patch(patch)


def local_first(ax):
  for i in range(len(ax)):
    ax[i].axis('off')

def local(figure, ax, i, per):
  if per == 100:
    img0 = np.asarray(Image.open('irisshap/shap'+ str(per)+str(i)+ '0.png'))
    ax[0].imshow(img0)
    img1 = np.asarray(Image.open('irisshap/shap'+ str(per)+str(i)+ '1.png'))
    ax[1].imshow(img1)
    img2 = np.asarray(Image.open('irisshap/shap'+ str(per)+str(i)+ '2.png'))
    ax[2].imshow(img2)
  else:
    img0 = np.asarray(Image.open('irisshap/shap0'+ str(per)+str(i)+ '0.png'))
    ax[0].imshow(img0)
    img1 = np.asarray(Image.open('irisshap/shap0'+ str(per)+str(i)+ '1.png'))
    ax[1].imshow(img1)
    img2 = np.asarray(Image.open('irisshap/shap0'+ str(per)+str(i)+ '2.png'))
    ax[2].imshow(img2)
  
  figure.canvas.draw_idle()

def visvar(glyph_width, glyph_height, pro, label, i_, cmap, true_y, dataset, axes, axl, alpha, base, per):
  global min_x
  global max_y
  global w
  global h   
  max_glyph_size = max(glyph_width, glyph_height)
  max_coordinates = np.amax(pro, axis=0)
  min_coordinates = np.amin(pro, axis=0)
  min_label = label.min()
  max_label = label.max()
  norm = matplotlib.colors.Normalize(vmin=min_label, vmax=max_label)
  color_map = matplotlib.colormaps.get_cmap(cmap)
  min_x = min_coordinates[0] - max_glyph_size
  max_x = max_coordinates[0] + max_glyph_size
  min_y = min_coordinates[1] - max_glyph_size
  max_y = max_coordinates[1] + max_glyph_size
  w = max_x-min_x
  h = max_y-min_y
  max_i = np.max(i_)
  axl.axis([min_x-5,
            min_x,
            0,
            20])
  axl.grid(False)
  axl.axis('off')

  axl.add_patch(Rectangle((min_x-5, 0), 5,20,facecolor='whitesmoke'))
  axl.text(min_x-4.75, 19, 'Percentage instances visible', ha='left', va='top', fontsize=8)
  if(per==25):
    axl.text(min_x-4.5, 18, '25%',  backgroundcolor='lightsteelblue', ha='left', va='top', fontsize=7)    
  else:
    axl.text(min_x-4.5, 18, '25%',  backgroundcolor='lightgrey', ha='left', va='top', fontsize=7) 
  if(per==50):
    axl.text(min_x-3.5, 18, '50%', backgroundcolor='lightsteelblue', ha='left', va='top',fontsize=7)   
  else:
    axl.text(min_x-3.5, 18, '50%', backgroundcolor='lightgrey', ha='left', va='top',fontsize=7) 
  if(per==75):
    axl.text(min_x-2.5, 18, '75%', backgroundcolor='lightsteelblue', ha='left', va='top',fontsize=7)   
  else:
    axl.text(min_x-2.5, 18, '75%', backgroundcolor='lightgrey', ha='left', va='top',fontsize=7)  
  if(per==100):
    axl.text(min_x-1.5, 18, '100%', backgroundcolor='lightsteelblue', ha='left', va='top',fontsize=7)   
  else:
    axl.text(min_x-1.5, 18, '100%', backgroundcolor='lightgrey', ha='left', va='top',fontsize=7)

  axl.add_patch(Rectangle((min_x-4.5, 10), 2, 0.5, facecolor='#377eb8', edgecolor='black',lw=0.5, alpha=0.8))
  axl.text(min_x-2.4, 10, 'Setosa', ha='left', va='bottom',fontsize=6)
  axl.add_patch(Rectangle((min_x-4.5, 9), 2, 0.5, facecolor='#4daf4a', edgecolor='black',lw=0.5, alpha=0.8))
  axl.text(min_x-2.4, 9, 'Versicolor', ha='left', va='bottom',fontsize=6)
  axl.add_patch(Rectangle((min_x-4.5, 8), 2, 0.5, facecolor='#984ea3', edgecolor='black',lw=0.5, alpha=0.8))
  axl.text(min_x-2.4, 8, 'Virginica', ha='left', va='bottom',fontsize=6)

  draw_porcupineglyph(min_x-2.5, 2.45, 2.5, [0,0,0], axl, 10, facecolor='white', alpha=1, base=1, i_=1, max_i=1)

  axl.text(min_x-1, 2.5, 'Setosa', ha='left', va='top',fontsize=6)
  axl.text(min_x-4, 4, 'Versicolor', ha='left', va='top',fontsize=6)
  axl.text(min_x-4, 1, 'Virginica', ha='left', va='top',fontsize=6)
  
  axes.axis([min_x,
              max_x,
              min_y,
              max_y])
  axes.grid(False)
  axes.axis('off')
  axes.set_aspect(1)
  for i in range(len(pro)):
    x_ = pro[i][0]
    y_ = pro[i][1]
    label_ = label[i]
    true_y_ = true_y[i]
    glyph_size_ = max_glyph_size
    draw_porcupineglyph(x_, y_, glyph_size_, dataset[i], axes, true_y=true_y_, alpha=alpha, facecolor=color_map(norm(label_)), base=base, i_=i_[i], max_i=max_i)
  return 

def visualize(pro_100, d_100, y_100, l_100, i_100, pro_75, d_75, y_75, l_75, i_75, pro_50, d_50, y_50, l_50, i_50, pro_25, d_25, y_25, l_25, i_25,
               glyph_width, glyph_height, base, names=None, cmap='Dark2', alpha=1.0, figsize=(5, 5), fontsize=6):
  figure = plt.figure(layout='constrained', figsize=figsize)
  gs = GridSpec(3, 3, height_ratios=[1,1,1], width_ratios=[1,5,2], wspace=0.01, hspace=0.01, figure=figure)
  axl = figure.add_subplot(gs[:,0])
  axes = figure.add_subplot(gs[:,1:-1])
  ax0 = figure.add_subplot(gs[0,-1])
  ax1 = figure.add_subplot(gs[1,-1])
  ax2 = figure.add_subplot(gs[2,-1])
  ax = [ax0,ax1,ax2]
  global per
  per=100

  visvar(glyph_width, glyph_height, pro_100, l_100, i_100, cmap, y_100, d_100, axes, axl, alpha, base, per)
  
  local_first(ax)

  def onclick(event):
    global min_x
    global max_y
    global w
    global h
    global per   
    if((event.xdata is not None) and (event.ydata is not None)):
      if(min_x-4.5 <= event.xdata < min_x-3.5 and  17<= event.ydata <18.5):
        axes.cla()
        axl.cla()
        ax[0].cla()
        ax[1].cla()
        ax[2].cla()
        plt.pause(0.001)
        local_first(ax)
        per = 25
        visvar(glyph_width, glyph_height, pro_25, l_25, i_25, cmap, y_25, d_25, axes, axl, alpha, base, per)
        plt.pause(0.001)
        figure.canvas.draw_idle()

      if(min_x-3.5 <= event.xdata < min_x-2.5 and 17<= event.ydata <18.5):
        axes.cla()
        axl.cla()
        ax[0].cla()
        ax[1].cla()
        ax[2].cla()
        plt.pause(0.001)
        local_first(ax)
        per = 50
        visvar(glyph_width, glyph_height, pro_50, l_50, i_50, cmap, y_50, d_50, axes, axl, alpha, base, per)       
        plt.pause(0.001)
        figure.canvas.draw_idle()

      if(min_x-2.5 <= event.xdata < min_x-1.5 and 17<= event.ydata <18.5):
        axes.cla()
        axl.cla()
        ax[0].cla()
        ax[1].cla()
        ax[2].cla()
        plt.pause(0.001)
        local_first(ax)
        per = 75
        visvar(glyph_width, glyph_height, pro_75, l_75, i_75, cmap, y_75, d_75, axes, axl, alpha, base, per)            
        plt.pause(0.001)
        figure.canvas.draw_idle()

      if(min_x-1.5 <= event.xdata < min_x-0.3 and 17<= event.ydata <18.5):
        axes.cla()
        axl.cla()
        ax[0].cla()
        ax[1].cla()
        ax[2].cla()
        plt.pause(0.001)
        local_first(ax)
        per=100
        visvar(glyph_width, glyph_height, pro_100, l_100, i_100, cmap, y_100, d_100, axes, axl, alpha, base, per)
        plt.pause(0.001)
        figure.canvas.draw_idle()
  
      x_clicked = math.floor((event.xdata+0.75)/1.5)*1.5
      y_clicked = math.floor((event.ydata+0.75)/1.5)*1.5
      if per == 100:
         slct = np.where((pro_100[:,0] == x_clicked) & (pro_100[:,1] == y_clicked))[0]
      elif per == 75:
         slct = np.where((pro_75[:,0] == x_clicked) & (pro_75[:,1] == y_clicked))[0]
      elif per == 50:
         slct = np.where((pro_50[:,0] == x_clicked) & (pro_50[:,1] == y_clicked))[0]
      elif per == 25:
         slct = np.where((pro_25[:,0] == x_clicked) & (pro_25[:,1] == y_clicked))[0]  

      if(slct.size!=0):
        local(figure,ax,slct[0], per)

  figManager = plt.get_current_fig_manager()
  figManager.window.showMaximized()
  cid = figure.canvas.mpl_connect('button_press_event', onclick)

def savefig(filename, dpi):
    plt.savefig(filename, dpi=dpi)

def show():
    plt.show()

def ml_model(X_tr, X_te, y_tr, y_te, lr, choice):
  if choice == 0:
    clf = GradientBoostingClassifier(n_estimators=20, learning_rate = lr, max_features=2, max_depth = 2, random_state = 0)
    clf.fit(X_tr,y_tr)
    explainer = shap.KernelExplainer(model=clf.predict_proba,data=X_tr, link='logit')
  elif choice == 1:
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_tr,y_tr)
    explainer = shap.Explainer(clf)
  elif choice == 2:
    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(X_tr,y_tr)
    explainer = shap.KernelExplainer(model=clf.predict_proba,data=X_tr, link='logit')

  y_pred = clf.predict(X_te)

  y_p = clf.predict_proba(X_te)

  y_prob =pd.DataFrame({
      'True y' : y_te,
      'prediction': y_pred
  })

  classes = len(y_p[0])

  for i in range(classes):
    y_prob['P of class '+str(i)] = y_p[:,i]

  return y_pred, y_p, y_prob, explainer

def reductionfactor(y, pred, y_test, sampsize = 1):
  y = y.astype(float)
  pred = pred.astype(float)
  y_test = y_test.astype(float)

  y_ = pd.DataFrame(y)
  pred_ = pd.DataFrame(pred)
  
  max_x = y_[0].max()+0.01
  min_x = y_[0].min()
  max_y = y_[1].max()+0.01
  min_y = y_[1].min()

  width = max_x-min_x
  height = max_y-min_y
  size = 0

  for i in range(sampsize):
      t1 = (min_x+(i*width/sampsize))
      t2 = (min_x+((i+1)*width/sampsize))
      for j in range(sampsize):
          t3 = (min_y+(j*height/sampsize))
          t4 = (min_y+((j+1)*height/sampsize))
          for k in range(3):
              if(len(pred_[(y_[0] >= t1) & (y_[0] < t2) & (y_[1] >= t3) & (y_[1] < t4)&(y_test==k)][0])>0):
                  t5 = pred_[(y_[0] >= t1) & (y_[0] < t2) & (y_[1] >= t3) & (y_[1] < t4)& (y_test==k)]
                  t6 = np.unique(t5, axis=0, return_counts=True)
                  size = size + t6[0].shape[0]

  reduction = size/len(y_test)*100
  return reduction

def sampling(y, pred, y_test, x_te, gb_ex, rf_ex, svm_ex, sampsize, redf, glyph_size, delta, gb_y_pred, rf_y_pred, svm_y_pred):
  y = y.astype(float)
  pred = pred.astype(float)
  y_test = y_test.astype(float)
  x_test = x_te.to_numpy()

  y_ = pd.DataFrame(y)
  pred_ = pd.DataFrame(pred)
  y_te = pd.DataFrame(y_test)
  
  max_x = y_[0].max()+0.01
  min_x = y_[0].min()
  max_y = y_[1].max()+0.01
  min_y = y_[1].min()

  width = max_x-min_x
  height = max_y-min_y
  pred_new = np.empty(shape=(0, 3))
  y_new = np.empty(shape=(0,2))
  y_te_new = np.empty(shape=(0))
  x_te_new = np.empty(shape=(0,4))
  items = np.empty(shape=(0))
  gb_y_pred_new = np.empty(shape=(0))
  rf_y_pred_new = np.empty(shape=(0))
  svm_y_pred_new = np.empty(shape=(0))

  for i in range(sampsize):
      t1 = (min_x+(i*width/sampsize))
      t2 = (min_x+((i+1)*width/sampsize))
      for j in range(sampsize):
          t3 = (min_y+(j*height/sampsize))
          t4 = (min_y+((j+1)*height/sampsize))
          for k in range(3):
              if(len(pred_[(y_[0] >= t1) & (y_[0] < t2) & (y_[1] >= t3) & (y_[1] < t4)&(y_test==k)][0])>0):
                  t5 = pred_[(y_[0] >= t1) & (y_[0] < t2) & (y_[1] >= t3) & (y_[1] < t4)& (y_test==k)]
                  t6 = np.unique(t5, axis=0, return_counts=True)
                  pred_new = np.append(pred_new,t6[0])
                  for l in range(len(t6[0])):
                      t7 = y_[(pred_[0]==t6[0][l][0]) &(pred_[1]==t6[0][l][1])& (pred_[2]==t6[0][l][2])&(y_[0] >= t1) & (y_[0] < t2) & (y_[1] >= t3) & (y_[1] < t4)&(y_test==k)].to_numpy()                    
                      coord = (np.average(t7[:,0]),np.average(t7[:,1]))
                      y_new = np.append(y_new, coord)
                      t8 = y_te[(pred_[0]==t6[0][l][0]) &(pred_[1]==t6[0][l][1])& (pred_[2]==t6[0][l][2])&(y_[0] >= t1) & (y_[0] < t2) & (y_[1] >= t3) & (y_[1] < t4)&(y_test==k)].to_numpy()
                      y_te_new=np.append(y_te_new,t8[0])
                      t9 = x_test[(pred_[0]==t6[0][l][0]) &(pred_[1]==t6[0][l][1])& (pred_[2]==t6[0][l][2])&(y_[0] >= t1) & (y_[0] < t2) & (y_[1] >= t3) & (y_[1] < t4)&(y_test==k)]
                      items = np.append(items,t9.shape[0])
                      x_te_new = np.append(x_te_new,None)
                      x_te_new[-1] = t9
                      t10 = gb_y_pred[(pred_[0]==t6[0][l][0]) &(pred_[1]==t6[0][l][1])& (pred_[2]==t6[0][l][2])&(y_[0] >= t1) & (y_[0] < t2) & (y_[1] >= t3) & (y_[1] < t4)&(y_test==k)]
                      gb_y_pred_new = np.append(gb_y_pred_new,t10[0])
                      t11 = rf_y_pred[(pred_[0]==t6[0][l][0]) &(pred_[1]==t6[0][l][1])& (pred_[2]==t6[0][l][2])&(y_[0] >= t1) & (y_[0] < t2) & (y_[1] >= t3) & (y_[1] < t4)&(y_test==k)]
                      rf_y_pred_new = np.append(rf_y_pred_new,t11[0])
                      t12 = svm_y_pred[(pred_[0]==t6[0][l][0]) &(pred_[1]==t6[0][l][1])& (pred_[2]==t6[0][l][2])&(y_[0] >= t1) & (y_[0] < t2) & (y_[1] >= t3) & (y_[1] < t4)&(y_test==k)]
                      svm_y_pred_new = np.append(svm_y_pred_new,t12[0])

  pred_new = np.reshape(pred_new,(int(len(pred_new)/3),3))
  y_new = np.reshape(y_new,(int(len(y_new)/2),2))

  # c_names = ['Setosa', 'Versicolor', 'Virginica']
  # for i in range(len(y_te_new)):
  #   shap_values = gb_ex.shap_values(x_te_new[i], silent=True)
  #   shap.summary_plot(shap_values[int(y_te_new[i])], x_te_new[i],show=False,feature_names=x_te.columns)
  #   axes = plt.gca()
  #   y_range = axes.get_ylim()
  #   x_range = axes.get_xlim()
  #   x_mid = (x_range[0]+x_range[1])/2
  #   plt.text(x_mid, y_range[1], 'Model 1     True='+ c_names[int(y_te_new[i])] + '     Predicted='+ c_names[int(gb_y_pred_new[i])], va='top', ha='center', color="black", fontsize=10)
  #   plt.savefig('irisshap/shap0'+ str(redf) + str(i)+'0')
  #   plt.close()
  #   shap_values = rf_ex.shap_values(x_te_new[i])
  #   shap.summary_plot(shap_values[int(y_te_new[i])], x_te_new[i],show=False,feature_names=x_te.columns)
  #   axes = plt.gca()
  #   y_range = axes.get_ylim()
  #   x_range = axes.get_xlim()
  #   x_mid = (x_range[0]+x_range[1])/2
  #   plt.text(x_mid, y_range[1], 'Model 2     True='+ c_names[int(y_te_new[i])] + '     Predicted='+ c_names[int(rf_y_pred_new[i])], va='top', ha='center', color="black", fontsize=10)
  #   plt.savefig('irisshap/shap0'+ str(redf) + str(i)+'1')
  #   plt.close()
  #   shap_values = svm_ex.shap_values(x_te_new[i], silent=True)
  #   shap.summary_plot(shap_values[int(y_te_new[i])], x_te_new[i],show=False,feature_names=x_te.columns)
  #   axes = plt.gca()
  #   y_range = axes.get_ylim()
  #   x_range = axes.get_xlim()
  #   x_mid = (x_range[0]+x_range[1])/2
  #   plt.text(x_mid, y_range[1], 'Model 3     True='+ c_names[int(y_te_new[i])] + '     Predicted='+ c_names[int(svm_y_pred_new[i])], va='top', ha='center', color="black", fontsize=10)
  #   plt.savefig('irisshap/shap0'+ str(redf) + str(i)+'2')
  #   plt.close()

  temp_y = y_te_new.copy()
  labels = None
  labels = y_te_new.copy()

  for i in range(len(temp_y)):
    wrong=0
    if(temp_y[i]==0 and (pred_new[i][0]<pred_new[i][1] or pred_new[i][0]<pred_new[i][2])):
      labels[i]=-1
    elif(temp_y[i]==1 and (pred_new[i][1]<pred_new[i][0] or pred_new[i][1]<pred_new[i][2])):
        labels[i]=-1
    elif(temp_y[i]==2 and (pred_new[i][2]<pred_new[i][0] or pred_new[i][2]<pred_new[i][1])):
        labels[i]=-1

  max_0 = np.max(y_new[:,0])
  max_1 = np.max(y_new[:,1])
  for i in range(len(y_new)):
     y_new[i,0] = y_new[i,0]/max_0
     y_new[i,1] = y_new[i,1]/max_1 
     
  y_new = DGrid(glyph_width=glyph_size, glyph_height=glyph_size, delta=delta).fit_transform(y_new)
  # move to glyph sizes 
  for i in range(len(y_new)):
    y_new[i,0] = round(y_new[i,0] - (y_new[i,0]%1.5),1)
    y_new[i,1] = round(y_new[i,1] - (y_new[i,1]%1.5),1)

  y_te_new = y_te_new.tolist()

  return y_new, pred_new, y_te_new, labels, items

def samphun(y, pred, y_test, x_te, gb_ex, rf_ex, svm_ex, redf, glyph_size, delta, gb_y_pred, rf_y_pred, svm_y_pred):
  x_test = x_te.to_numpy()

  # c_names = ['Setosa', 'Versicolor', 'Virginica']
  # for i in range(len(y_test)):
  #   x_temp = np.asarray([x_test[i]])
  #   shap_values = gb_ex.shap_values(x_temp, silent=True)
  #   shap.summary_plot(shap_values[int(y_test[i])], x_temp,show=False,feature_names=x_te.columns)
  #   axes = plt.gca()
  #   y_range = axes.get_ylim()
  #   x_range = axes.get_xlim()
  #   x_mid = (x_range[0]+x_range[1])/2
  #   plt.text(x_mid, y_range[1], 'Model 1     True='+ c_names[int(y_test[i])] + '     Predicted='+ c_names[int(gb_y_pred[i])], va='top', ha='center', color="black", fontsize=10)
  #   plt.savefig('irisshap/shap'+ str(redf) + str(i)+'0')
  #   plt.close()
  #   shap_values = rf_ex.shap_values(x_temp)
  #   shap.summary_plot(shap_values[int(y_test[i])], x_temp,show=False,feature_names=x_te.columns)
  #   axes = plt.gca()
  #   y_range = axes.get_ylim()
  #   x_range = axes.get_xlim()
  #   x_mid = (x_range[0]+x_range[1])/2
  #   plt.text(x_mid, y_range[1], 'Model 2     True='+ c_names[int(y_test[i])] + '     Predicted='+ c_names[int(rf_y_pred[i])], va='top', ha='center', color="black", fontsize=10)
  #   plt.savefig('irisshap/shap'+ str(redf) + str(i)+'1')
  #   plt.close()
  #   shap_values = svm_ex.shap_values(x_temp, silent=True)
  #   shap.summary_plot(shap_values[int(y_test[i])], x_temp,show=False,feature_names=x_te.columns)
  #   axes = plt.gca()
  #   y_range = axes.get_ylim()
  #   x_range = axes.get_xlim()
  #   x_mid = (x_range[0]+x_range[1])/2
  #   plt.text(x_mid, y_range[1], 'Model 3     True='+ c_names[int(y_test[i])] + '     Predicted='+ c_names[int(svm_y_pred[i])], va='top', ha='center', color="black", fontsize=10)
  #   plt.savefig('irisshap/shap'+ str(redf) + str(i)+'2')
  #   plt.close()
  
  temp_y = y_test.copy()
  labels = None
  labels = y_test.copy()

  for i in range(len(temp_y)):
    wrong=0
    if(temp_y[i]==0 and (pred[i][0]<pred[i][1] or pred[i][0]<pred[i][2])):
      labels[i]=-1
    elif(temp_y[i]==1 and (pred[i][1]<pred[i][0] or pred[i][1]<pred[i][2])):
        labels[i]=-1
    elif(temp_y[i]==2 and (pred[i][2]<pred[i][0] or pred[i][2]<pred[i][1])):
        labels[i]=-1
  
  max_0 = np.max(y[:,0])
  max_1 = np.max(y[:,1])
  for i in range(len(y)):
     y[i,0] = y[i,0]/max_0
     y[i,1] = y[i,1]/max_1 
     
  y_new = DGrid(glyph_width=glyph_size, glyph_height=glyph_size, delta=delta).fit_transform(y)
  # move to glyph sizes 
  for i in range(len(y_new)):
    y_new[i,0] = round(y_new[i,0] - (y_new[i,0]%1.5),1)
    y_new[i,1] = round(y_new[i,1] - (y_new[i,1]%1.5),1)

  y_test = y_test.tolist()

  items = np.full(len(y_test), 1)
  return y_new, pred, y_test, labels, items

iris = datasets.load_iris()
data=pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})

X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y=data['species']  # Labels

state = 12
test_size = 0.8

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=state)

gb_y_pred,  gb_y_p,  gb_y_prob,  gb_ex  = ml_model(X_train,X_test,y_train,y_test, 0.05, 0)
rf_y_pred,  rf_y_p,  rf_y_prob,  rf_ex  = ml_model(X_train,X_test,y_train,y_test, 0, 1)
svm_y_pred, svm_y_p, svm_y_prob, svm_ex = ml_model(X_train,X_test,y_train,y_test, 0, 2)


instances = len(y_test)
voting_pred = np.zeros((3, instances))

for i in range(instances):
  voting_pred[gb_y_pred[i]][i] += 1
  voting_pred[rf_y_pred[i]][i] += 1
  voting_pred[svm_y_pred[i]][i] += 1

voting_tot= pd.DataFrame({
    'True y' : y_test
})
for i in range(3):
    voting_tot['Votes class '+str(i)] = voting_pred[i]

X_orig = X_test.to_numpy()
X = preprocessing.StandardScaler().fit_transform(X_test.to_numpy())

# apply dimensionality reduction
umap = UMAP(n_components=2, init='random', random_state=0)
y = umap.fit_transform(X)
preds=voting_tot.iloc[:,1:].to_numpy()
preds=preds/3

# red = 0
# ssize = 1
# reds = np.empty(shape=(0))
# s25 = 0
# s50 = 0
# s75 = 0
# while red<75 and ssize<100:
#   red = reductionfactor(y, preds, y_test.to_numpy(), sampsize=ssize)
#   if(len(reds)>0):
#     if(reds[-1]<25 and red>25):
#       s25=ssize
#     elif(reds[-1]<50 and red>50):
#        s50=ssize
#     elif(reds[-1]<75 and red>75):
#        s75=ssize
#   reds = np.append(reds,red)
#   ssize = ssize+1

s25=9
s50=21
s75=40

glyph_size = 1.5

y_s25,  pred_s25,  y_te_s25,  labels_s25,  i_s25  = sampling(y, preds, y_test.to_numpy(), X_test, gb_ex, rf_ex, svm_ex, s25, 25, glyph_size, 200, gb_y_pred, rf_y_pred, svm_y_pred)
y_s50,  pred_s50,  y_te_s50,  labels_s50,  i_s50  = sampling(y, preds, y_test.to_numpy(), X_test, gb_ex, rf_ex, svm_ex, s50, 50, glyph_size, 300, gb_y_pred, rf_y_pred, svm_y_pred)
y_s75,  pred_s75,  y_te_s75,  labels_s75,  i_s75  = sampling(y, preds, y_test.to_numpy(), X_test, gb_ex, rf_ex, svm_ex, s75, 75, glyph_size, 400, gb_y_pred, rf_y_pred, svm_y_pred)
y_s100, pred_s100, y_te_s100, labels_s100, i_s100 = samphun(y,  preds, y_test.to_numpy(), X_test, gb_ex,rf_ex, svm_ex,      100, glyph_size, 500, gb_y_pred, rf_y_pred, svm_y_pred)

clrs=['r','#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
cust_cmap = clr.ListedColormap(clrs[0:4])
width = 30
height =20

print("--- Visbuild execution %s seconds ---" % (time.time() - start_time))

min_x = 0
max_y = 0
w = 0
h = 0   
per = 100

visualize(y_s100, pred_s100, y_te_s100, labels_s100, i_s100, y_s75, pred_s75, y_te_s75, labels_s75, i_s75, 
          y_s50, pred_s50, y_te_s50, labels_s50, i_s50, y_s25, pred_s25,  y_te_s25, labels_s25, i_s25, 
          glyph_width=glyph_size, glyph_height=glyph_size, cmap=cust_cmap, figsize=(width, height), alpha=0.8, base=0.3)

show()