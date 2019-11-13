import matplotlib.pyplot as plt
import numpy as np

# gating_hypparams = dict(K=nb_models, gammas=np.ones((nb_models, )), deltas=np.ones((nb_models, ))*1)
# gating_prior = distributions.StickBreaking(**gating_hypparams)

# # define components
# components_hypparams = dict(mu=np.zeros((in_dim_niw,)),
# # components_hypparams = dict(mu=npr.uniform(-10,40,size=in_dim_niw),
#                  kappa=0.05, #0.05
#                  psi_niw=np.eye(in_dim_niw),
#                  nu_niw=2 * in_dim_niw + 1,
#                  M=np.zeros((out_dim, in_dim_mniw)),
#                  # V=10. * np.eye(in_dim_mniw),
#                  V= np.asarray([[1, 0],[0, 30]]),
#                  affine=affine,
#                  psi_mniw=np.eye(out_dim) * 0.01,
#                  nu_mniw=2 * out_dim + 1)
# components_prior = distributions.NormalInverseWishartMatrixNormalInverseWishart(**components_hypparams)


# 3-dimensional plot of prediction
# fig = plt.figure()
# ax = Axes3D(fig)
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(data[:, 0], data[:, 1], data[:, 2], 'kx', zorder=1)
# ax.scatter(data[:, 0], data[:, 1], pred_y, c='red', s=3, zorder=2)
# # ax.title('best model')
# plt.show()

# prediction
# componentA = np.zeros([len(gmm.components),np.size(gmm.components[0].A)])
# # componentA = np.array([len(gmm.components),(gmm.components[0].A.shape[0],gmm.components[0].A.shape[1])])
# # component_mu = np.zeros([len(gmm.components),np.size(gmm.components[0].mu)])
# print(componentA.shape)
# for idx, component in enumerate(gmm.components):
#     print(component.A.shape)
#     componentA[idx] = component.A
#     # component_mu[idx] = component.mu


# pred_y = np.zeros((data.shape[0], out_dim))
# for i in range(data.shape[0]):
#     idx = gmm.labels_list[0].z[i]
#     pred_y[i] = gmm.components[idx].predict(data[i,:-out_dim])
#     # if affine:
#     #     pred_y[i] = np.matmul(componentA[idx,:-1], data[i,:-out_dim].T) + componentA[idx,in_dim_niw:] * 1
#     # else:
#     #     pred_y[i] = np.matmul(componentA[idx,:], data[i,:-out_dim].T)

# f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
#
# ax1.plot(range(10),2*np.arange(10))
# ax2.plot(range(10),range(10))
# ax3.plot(range(10),range(10))
# ax4.plot(range(10),range(10))
# # ax3.plot(range(5),np.arange(5)*1000)
#
# plt.show()

# # term = term + mat_T * marg[idx] / marg.sum() * alphas[idx] / alphas_hat# *len(dpglm.used_labels)#marg.sum() #  FIXME version 1: der theorie entsprechend mit alphas, problem: nb_models nicht in theorie und kleine modelle verursachen probleme
# term = term + mat_T * marg[idx] / marg.sum()
# # fixme version 2: ohne alphas und nb_models. besser, aber nicht theorie entsprechend (alphas weg)
# # term = term + mat_T * (marg[idx] / marg.sum() + alphas[idx] / alphas_hat) / 2
# # alternative 3