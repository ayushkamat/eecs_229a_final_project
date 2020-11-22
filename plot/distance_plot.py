import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap


fname = {FNAME}
home_dir = {HOMEDIR}

with open(fname, 'rb') as f:
    data = pickle.load(f)

for item in data:
    for key in item.keys():
        if str(type(item[key])).find('torch') >= 0:
            item[key] = item[key].item()

df = pd.DataFrame.from_dict(data)
sns_plot = sns.lmplot(x='kldiv',y='test/acc',data=df,fit_reg=False, hue='teacher', legend=False)
sns.regplot(x="kldiv", y="test/acc", data=df, scatter=False, ax=sns_plot.axes[0, 0])
sns_plot.savefig(home_dir+'/{0}.png'.format('distance_gmms'))

sns.set()
sns_plot = sns.lmplot(x='test/teacher_acc',y='test/acc',data=df,fit_reg=False, hue='teacher', legend=False)
sns.regplot(x="test/teacher_acc", y="test/acc", data=df, scatter=False, ax=sns_plot.axes[0, 0])
sns_plot.savefig(home_dir+'/{0}.png'.format('distanc_acc_gmms'))

sns.set()

elev = 18.0
azim = -60

fig = plt.figure(figsize=(6,6))
ax = Axes3D(fig)
cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())
plt.xlim(right=60)
sc = ax.scatter(df['kldiv'], df['test/teacher_acc'], df['test/acc'], marker='o', cmap=cmap, alpha=1, c=df['teacher'])
ax.view_init(elev=elev, azim=azim)
ax.set_xlabel('kldiv')
ax.set_ylabel('teacher_acc')
ax.set_zlabel('student_acc')
plt.savefig(home_dir+'/3d_distance.png', bbox_inches='tight')

fig = plt.figure(figsize=(6,6))
ax = Axes3D(fig)
cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())
plt.xlim(right=60)
sc = ax.scatter(df['kldiv'], df['test/teacher_acc'], df['test/acc'], marker='o', cmap=cmap, alpha=1)
ax.view_init(elev=elev, azim=azim)
ax.set_xlabel('kldiv')
ax.set_ylabel('teacher_acc')
ax.set_zlabel('student_acc')
plt.savefig(home_dir+'/3d_distance_nocolor.png', bbox_inches='tight')

