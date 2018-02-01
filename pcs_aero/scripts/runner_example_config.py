from pcs_aero.experiment import ExperimentGroup

experiments = [
#     ExperimentGroup(
#         'results/r92R{i}T60.txt'.format(i=i), {
#             'shape':
#             ['circle', 'rect', 'halfcircle', 'triangle', 'moon', 'bullet'],
#             'r': [92],
#             'time': [60],
#             'Re': [i],
#             'theta':
#             range(0, 180, 1)
#         }) for i in [*range(55, 66), *range(95, 105)]
# ] + [
#     ExperimentGroup('results/reynolds.txt', {
#         'shape': ['circle', 'rect', 'triangle'],
#         'time': [40],
#         'Re': np.linspace(17, 18, 5)
#     }),
#     ExperimentGroup('results/velocity.txt', {
#         'shape':
#         ['circle', 'rect', 'halfcircle', 'triangle', 'moon', 'bullet'],
#         'Uin':
#         np.arange(1, 16) / 100,
#         'time': [180],
#         'Re': [17.5],
#     }),
#     ExperimentGroup('results/size.txt', {
#         'shape':
#         ['circle', 'rect', 'halfcircle', 'triangle', 'moon', 'bullet'],
#         'size':
#         [1 / 32, 1 / 24, 1 / 16, 1 / 12, 1 / 8, 1 / 6, 1 / 4, 1 / 3],
#         'time': [40],
#         'Re': [17.5],
#     }),
#     ExperimentGroup('results/csize.txt', {
#         'shape': ['circle'],
#         'size': np.linspace(1 / 32, 0.6, 100),
#         'time': [40],
#         'Re': [17.5],
#     }),
    ExperimentGroup('example.txt', {
        'shape': ['bullet'],
        'theta': range(0, 16, 5),
    })
]