# # -------------------- iterations ---------------------------------------

# from itertools import permutations, product
# list( permutations(['A','B','C'], 2) ) # returns: [('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A'), ('C', 'B')]
# list( product(['A','B'], range(3)) ) # returns: [('A', 0), ('A', 1), ('A', 2), ('B', 0), ('B', 1), ('B', 2)]
# list( zip(['A','B','C'], range(3)) ) # returns: [('A', 0), ('B', 1), ('C', 2)]

# # --------------------------- plotting ---------------------------------

# import matplotlib.pyplot as plt
# fig,ax = plt.subplots()
# ax.plot(x,np.sin(x),'r.--') # fmt = '[color][marker][line]'
# ax.set(xlim=(-1,1), ylim=(-1,1), xlabel='xlabel', ylabel='ylabel', title='title')
# plt.savefig('plots/test.pdf')
# plt.show()

# # ----------------------- create animations ------------------------------------

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, PillowWriter, HTMLWriter

# # Parameters
# num_frames, t_end = 100, 5
# frames = np.linspace(0, t_end, num_frames)
# fps = num_frames / t_end

# fig, ax = plt.subplots()
# ln, = ax.plot([], [], 'r-', animated=True)
# ax.set(xlim=(0,5), ylim=(-1,1), xlabel='X', ylabel='Y', title='Sine Wave Animation')

# def update(frame):
#     xdata = np.linspace(0, 5, 100)
#     ydata = np.sin(xdata + frame)
#     ln.set_data(xdata, ydata)
#     return ln,

# ani = FuncAnimation(fig, update, frames=frames, blit=True)
# ani.save('animations/sine.gif', writer=PillowWriter(fps=0.5*fps)) # save animation as .gif
# # ani.save('animations/sine.html', writer=HTMLWriter(fps=0.5*fps)) # save animation as .html 
# # this additionally creates a folder with all frames saved as .png
# plt.close()

# # ----------------------- magic commands ----------------------------------

# # create test.py with:
# if __name__=='__main__':
#     print('Hello World')
# # line magics: %, cell magics: %%
# %run test.py # runs a python script
# %whos # shows all global variables 
# %reset # deletes global variables 
# %%latex # enter latex code, e.g., $a^2+b^2=c^2$ (because of two % the input must be in the next line)
# %pwd # returns the current working directory
# %time # measured the time for the executing of the following function








