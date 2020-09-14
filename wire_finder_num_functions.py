import numpy as np

def sort(array):
    array_to_sort = np.zeros((len(array), 1), np.uint32)
    for i in range(len(array)):
        array_to_sort[i] = array[i, 0] * array[i, 1]
    idx = np.argsort(array_to_sort, 0).flatten()
    new_array = array[idx]
    return new_array

def find_extremes(cnt):
    extrema = np.zeros((4, 2))
    extrema[0, :] = tuple(cnt[0][cnt[0][:, :, 0].argmin()][0])  # leftmost
    extrema[1, :] = tuple(cnt[0][cnt[0][:, :, 0].argmax()][0])  # rightmost
    extrema[2, :] = tuple(cnt[0][cnt[0][:, :, 1].argmin()][0])  # topmost
    extrema[3, :] = tuple(cnt[0][cnt[0][:, :, 1].argmax()][0])  # bottommost
    return extrema

def vertical_contact(p_contact, n_contact, lead, angle, orientation_path):
    first_lead = np.zeros([9, 2])
    second_lead = np.zeros([9, 2])
    if p_contact[0] < n_contact[0]:
        if orientation_path == 'up':
            lead_index = np.array([0, 1])
            if angle < 0:
                first_lead[0, :] = [p_contact[0], p_contact[1] - 4.3301]
                if p_contact[0] < lead[lead_index[0], 0] - 10:  # checks if extremity is on the left of the lead
                    first_lead[1, :] = [first_lead[0, 0] + 20, first_lead[0, 1]]
                    first_lead[2, :] = [first_lead[1, 0], lead[lead_index[0], 1]]
                    first_lead[3, :] = [lead[lead_index[0], 0] + 10, first_lead[2, 1]]
                    first_lead[4, :] = [first_lead[3, 0], first_lead[3, 1] + 10]
                    first_lead[5, :] = [first_lead[1, 0] - 10, first_lead[4, 1]]
                    first_lead[6, :] = [first_lead[5, 0], first_lead[0, 1] + 8.6602]
                else:  # if wire is not on the left, it is on the right
                    first_lead[1, :] = [first_lead[0, 0] + 20, first_lead[0, 1]]
                    first_lead[2, :] = [first_lead[1, 0], lead[lead_index[0], 1] + 10]
                    first_lead[3, :] = [lead[lead_index[0], 0] - 10, first_lead[2, 1]]
                    first_lead[4, :] = [first_lead[3, 0], first_lead[3, 1] - 10]
                    first_lead[5, :] = [first_lead[1, 0] - 10, first_lead[4, 1]]
                    first_lead[6, :] = [first_lead[5, 0], first_lead[0, 1] + 8.6602]
                first_lead[7, :] = [first_lead[0, 0], first_lead[6, 1]]
                first_lead[8, :] = first_lead[0, :]
                # second_lead Coordinates
                second_lead[0, :] = [n_contact[0], n_contact[1] - 4.3301]
                if n_contact[0] < lead[lead_index[1], 0] + 10:
                    second_lead[1, :] = [second_lead[0, 0] - 10, second_lead[0, 1]]
                    second_lead[2, :] = [second_lead[1, 0], lead[lead_index[1], 1]]
                    second_lead[3, :] = [lead[lead_index[1], 0] + 10, second_lead[2, 1]]
                    second_lead[4, :] = [second_lead[3, 0], second_lead[3, 1] - 10]
                    second_lead[5, :] = [second_lead[1, 0] - 10, second_lead[4, 1]]
                    second_lead[6, :] = [second_lead[5, 0], second_lead[0, 1] + 8.6602]
                else:
                    second_lead[1, :] = [second_lead[0, 0] - 10, second_lead[0, 1]]
                    second_lead[2, :] = [second_lead[1, 0], lead[lead_index[1], 1] - 10]
                    second_lead[3, :] = [lead[lead_index[1], 0] - 10, second_lead[2, 1]]
                    second_lead[4, :] = [second_lead[3, 0], second_lead[3, 1] + 10]
                    second_lead[5, :] = [second_lead[1, 0] - 10, second_lead[4, 1]]
                    second_lead[6, :] = [second_lead[5, 0], second_lead[0, 1] + 8.6602]
                second_lead[7, :] = [second_lead[0, 0], second_lead[6, 1]]
                second_lead[8, :] = second_lead[0, :]
            else:  # angle of wire is positive[ / ]
                first_lead[0, :] = [p_contact[0], p_contact[1] - 4.3301]
                if p_contact[0] < lead[lead_index[0], 0] + 10:  # checks if extremity is on the left of the lead
                    first_lead[1, :] = [first_lead[0, 0] - 20, first_lead[0, 1]]
                    first_lead[2, :] = [first_lead[1, 0], lead[lead_index[0], 1] + 10]
                    first_lead[3, :] = [lead[lead_index[0], 0] + 10, first_lead[2, 1]]
                    first_lead[4, :] = [first_lead[3, 0], first_lead[3, 1] - 10]
                    first_lead[5, :] = [first_lead[1, 0] + 10, first_lead[4, 1]]
                    first_lead[6, :] = [first_lead[5, 0], first_lead[0, 1] + 8.6602]
                else:  # if wire is not on the left, it is on the right
                    first_lead[1, :] = [first_lead[0, 0] - 20, first_lead[0, 1]]
                    first_lead[2, :] = [first_lead[1, 0], lead[lead_index[0], 1]]
                    first_lead[3, :] = [lead[lead_index[0], 0] - 10, first_lead[2, 1]]
                    first_lead[4, :] = [first_lead[3, 0], first_lead[3, 1] + 10]
                    first_lead[5, :] = [first_lead[1, 0] + 10, first_lead[4, 1]]
                    first_lead[6, :] = [first_lead[5, 0], first_lead[0, 1] + 8.6602]
                first_lead[7, :] = [first_lead[0, 0], first_lead[6, 1]]
                first_lead[8, :] = first_lead[0, :]
                # second_lead Coordinates
                second_lead[0, :] = [n_contact[0], n_contact[1] - 4.3301]
                if n_contact[0] < lead[lead_index[1], 0] - 10:
                    second_lead[1, :] = [second_lead[0, 0] + 10, second_lead[0, 1]]
                    second_lead[2, :] = [second_lead[1, 0], lead[lead_index[1], 1] - 10]
                    second_lead[3, :] = [lead[lead_index[1], 0] + 10, second_lead[2, 1]]
                    second_lead[4, :] = [second_lead[3, 0], second_lead[3, 1] + 10]
                    second_lead[5, :] = [second_lead[1, 0] + 10, second_lead[4, 1]]
                    second_lead[6, :] = [second_lead[5, 0], second_lead[0, 1] + 8.6602]
                else:
                    second_lead[1, :] = [second_lead[0, 0] + 10, second_lead[0, 1]]
                    second_lead[2, :] = [second_lead[1, 0], lead[lead_index[1], 1]]
                    second_lead[3, :] = [lead[lead_index[1], 0] - 10, second_lead[2, 1]]
                    second_lead[4, :] = [second_lead[3, 0], second_lead[3, 1] - 10]
                    second_lead[5, :] = [second_lead[1, 0] + 10, second_lead[4, 1]]
                    second_lead[6, :] = [second_lead[5, 0], second_lead[0, 1] + 8.6602]
                second_lead[7, :] = [second_lead[0, 0], second_lead[6, 1]]
                second_lead[8, :] = second_lead[0, :]
        elif orientation_path == 'down':
            lead_index = np.array([1, 0])
            if angle < 0:  # Angle of the wire is negative [ \ ]
                first_lead[0, :] = [p_contact[0], p_contact[1] - 4.3301]
                if p_contact[0] < lead[lead_index[0], 0] - 10:
                    first_lead[1, :] = [first_lead[0, 0] + 10, first_lead[0, 1]]
                    first_lead[2, :] = [first_lead[1, 0], lead[lead_index[0], 1] - 10]
                    first_lead[3, :] = [lead[lead_index[0], 0] + 10, first_lead[2, 1]]
                    first_lead[4, :] = [first_lead[3, 0], first_lead[3, 1] + 10]
                    first_lead[5, :] = [first_lead[1, 0] + 10, first_lead[4, 1]]
                    first_lead[6, :] = [first_lead[5, 0], first_lead[0, 1] + 8.6602]
                else:
                    first_lead[1, :] = [first_lead[0, 0] + 10, first_lead[0, 1]]
                    first_lead[2, :] = [first_lead[1, 0], lead[lead_index[0], 1]]
                    first_lead[3, :] = [lead[lead_index[0], 0] - 10, first_lead[2, 1]]
                    first_lead[4, :] = [first_lead[3, 0], first_lead[3, 1] - 10]
                    first_lead[5, :] = [first_lead[1, 0] + 10, first_lead[4, 1]]
                    first_lead[6, :] = [first_lead[5, 0], first_lead[0, 1] + 8.6602]
                first_lead[7, :] = [first_lead[0, 0], first_lead[6, 1]]
                first_lead[8, :] = first_lead[0, :]
                # second_lead Coordinates
                second_lead[0, :] = [n_contact[0], n_contact[1] - 4.3301]
                if n_contact[0] < lead[lead_index[1], 0] + 10:
                    second_lead[1, :] = [second_lead[0, 0] - 20, second_lead[0, 1]]
                    second_lead[2, :] = [second_lead[1, 0], lead[lead_index[1], 1] + 10]
                    second_lead[3, :] = [lead[lead_index[1], 0] + 10, second_lead[2, 1]]
                    second_lead[4, :] = [second_lead[3, 0], second_lead[3, 1] - 10]
                    second_lead[5, :] = [second_lead[1, 0] + 10, second_lead[4, 1]]
                    second_lead[6, :] = [second_lead[5, 0], second_lead[0, 1] + 8.6602]
                else:
                    second_lead[1, :] = [second_lead[0, 0] - 20, second_lead[0, 1]]
                    second_lead[2, :] = [second_lead[1, 0], lead[lead_index[1], 1]]
                    second_lead[3, :] = [lead[lead_index[1], 0] - 10, second_lead[2, 1]]
                    second_lead[4, :] = [second_lead[3, 0], second_lead[3, 1] + 10]
                    second_lead[5, :] = [second_lead[1, 0] + 10, second_lead[4, 1]]
                    second_lead[6, :] = [second_lead[5, 0], second_lead[0, 1] + 8.6602]
                second_lead[7, :] = [second_lead[0, 0], second_lead[6, 1]]
                second_lead[8, :] = second_lead[0, :]
            else:  # angle of wire is positive[ / ]
                first_lead[0, :] = [p_contact[0], p_contact[1] - 4.3301]
                if p_contact[0] < lead[lead_index[0], 0] + 10:
                    first_lead[1, :] = [first_lead[0, 0] - 10, first_lead[0, 1]]
                    first_lead[2, :] = [first_lead[1, 0], lead[lead_index[0], 1]]
                    first_lead[3, :] = [lead[lead_index[0], 0] + 10, first_lead[2, 1]]
                    first_lead[4, :] = [first_lead[3, 0], first_lead[3, 1] - 10]
                    first_lead[5, :] = [first_lead[1, 0] - 10, first_lead[4, 1]]
                    first_lead[6, :] = [first_lead[5, 0], first_lead[0, 1] + 8.6602]
                else:
                    first_lead[1, :] = [first_lead[0, 0] - 10, first_lead[0, 1]]
                    first_lead[2, :] = [first_lead[1, 0], lead[lead_index[0], 1] - 10]
                    first_lead[3, :] = [lead[lead_index[0], 0] - 10, first_lead[2, 1]]
                    first_lead[4, :] = [first_lead[3, 0], first_lead[3, 1] + 10]
                    first_lead[5, :] = [first_lead[1, 0] - 10, first_lead[4, 1]]
                    first_lead[6, :] = [first_lead[5, 0], first_lead[0, 1] + 8.6602]
                first_lead[7, :] = [first_lead[0, 0], first_lead[6, 1]]
                first_lead[8, :] = first_lead[0, :]
                # second_lead Coordinates
                second_lead[0, :] = [n_contact[0], n_contact[1] - 4.3301]
                if n_contact[0] < lead[lead_index[1], 0] - 10:
                    second_lead[1, :] = [second_lead[0, 0] + 20, second_lead[0, 1]]
                    second_lead[2, :] = [second_lead[1, 0], lead[lead_index[1], 1]]
                    second_lead[3, :] = [lead[lead_index[1], 0] + 10, second_lead[2, 1]]
                    second_lead[4, :] = [second_lead[3, 0], second_lead[3, 1] + 10]
                    second_lead[5, :] = [second_lead[1, 0] - 10, second_lead[4, 1]]
                    second_lead[6, :] = [second_lead[5, 0], second_lead[0, 1] + 8.6602]
                else:
                    second_lead[1, :] = [second_lead[0, 0] + 20, second_lead[0, 1]]
                    second_lead[2, :] = [second_lead[1, 0], lead[lead_index[1], 1] + 10]
                    second_lead[3, :] = [lead[lead_index[1], 0] - 10, second_lead[2, 1]]
                    second_lead[4, :] = [second_lead[3, 0], second_lead[3, 1] - 10]
                    second_lead[5, :] = [second_lead[1, 0] - 10, second_lead[4, 1]]
                    second_lead[6, :] = [second_lead[5, 0], second_lead[0, 1] + 8.6602]
                second_lead[7, :] = [second_lead[0, 0], second_lead[6, 1]]
                second_lead[8, :] = second_lead[0, :]
    else:  # p extremity is up
        if orientation_path == "up":
            lead_index = np.array([0, 1])
            if angle < 0:  # Angle of the wire is negative [ \ ]
                first_lead[0, :] = [p_contact[0], p_contact[1] - 4.3301]
                if p_contact[0] < lead[lead_index[0], 0] + 10:  # checks if extremity is on the left of the lead
                    first_lead[1, :] = [first_lead[0, 0] - 20, first_lead[0, 1]]
                    first_lead[2, :] = [first_lead[1, 0], lead[lead_index[0], 1] + 10]
                    first_lead[3, :] = [lead[lead_index[0], 0] + 10, first_lead[2, 1]]
                    first_lead[4, :] = [first_lead[3, 0], first_lead[3, 1] - 10]
                    first_lead[5, :] = [first_lead[1, 0] + 10, first_lead[4, 1]]
                    first_lead[6, :] = [first_lead[5, 0], first_lead[0, 1] + 8.6602]
                else:  # if wire is not on the left, it is on the right
                    first_lead[1, :] = [first_lead[0, 0] - 20, first_lead[0, 1]]
                    first_lead[2, :] = [first_lead[1, 0], lead[lead_index[0], 1]]
                    first_lead[3, :] = [lead[lead_index[0], 0] - 10, first_lead[2, 1]]
                    first_lead[4, :] = [first_lead[3, 0], first_lead[3, 1] + 10]
                    first_lead[5, :] = [first_lead[1, 0] + 10, first_lead[4, 1]]
                    first_lead[6, :] = [first_lead[5, 0], first_lead[0, 1] + 8.6602]
                first_lead[7, :] = [first_lead[0, 0], first_lead[6, 1]]
                first_lead[8, :] = first_lead[0, :]
                # second_lead Coordinates
                second_lead[0, :] = [n_contact[0], n_contact[1] - 4.3301]
                if n_contact[0] < lead[lead_index[1], 0] - 10:
                    second_lead[1, :] = [second_lead[0, 0] + 10, second_lead[0, 1]]
                    second_lead[2, :] = [second_lead[1, 0], lead[lead_index[1], 1] - 10]
                    second_lead[3, :] = [lead[lead_index[1], 0] + 10, second_lead[2, 1]]
                    second_lead[4, :] = [second_lead[3, 0], second_lead[3, 1] + 10]
                    second_lead[5, :] = [second_lead[1, 0] + 10, second_lead[4, 1]]
                    second_lead[6, :] = [second_lead[5, 0], second_lead[0, 1] + 8.6602]
                else:
                    second_lead[1, :] = [second_lead[0, 0] + 10, second_lead[0, 1]]
                    second_lead[2, :] = [second_lead[1, 0], lead[lead_index[1], 1]]
                    second_lead[3, :] = [lead[lead_index[1], 0] - 10, second_lead[2, 1]]
                    second_lead[4, :] = [second_lead[3, 0], second_lead[3, 1] - 10]
                    second_lead[5, :] = [second_lead[1, 0] + 10, second_lead[4, 1]]
                    second_lead[6, :] = [second_lead[5, 0], second_lead[0, 1] + 8.6602]
                second_lead[7, :] = [second_lead[0, 0], second_lead[6, 1]]
                second_lead[8, :] = second_lead[0, :]
            else:  # angle of wire is positive[ / ]
                first_lead[0, :] = [p_contact[0], p_contact[1] - 4.3301]
                if p_contact[0] < lead[lead_index[0], 0] - 10:  # checks if extremity is on the left of the lead
                    first_lead[1, :] = [first_lead[0, 0] + 20, first_lead[0, 1]]
                    first_lead[2, :] = [first_lead[1, 0], lead[lead_index[0], 1]]
                    first_lead[3, :] = [lead[lead_index[0], 0] + 10, first_lead[2, 1]]
                    first_lead[4, :] = [first_lead[3, 0], first_lead[3, 1] + 10]
                    first_lead[5, :] = [first_lead[1, 0] - 10, first_lead[4, 1]]
                    first_lead[6, :] = [first_lead[5, 0], first_lead[0, 1] + 8.6602]
                else:  # if wire is not on the left, it is on the right
                    first_lead[1, :] = [first_lead[0, 0] + 20, first_lead[0, 1]]
                    first_lead[2, :] = [first_lead[1, 0], lead[lead_index[0], 1] + 10]
                    first_lead[3, :] = [lead[lead_index[0], 0] - 10, first_lead[2, 1]]
                    first_lead[4, :] = [first_lead[3, 0], first_lead[3, 1] - 10]
                    first_lead[5, :] = [first_lead[1, 0] - 10, first_lead[4, 1]]
                    first_lead[6, :] = [first_lead[5, 0], first_lead[0, 1] + 8.6602]
                first_lead[7, :] = [first_lead[0, 0], first_lead[6, 1]]
                first_lead[8, :] = first_lead[0, :]
                # second_lead Coordinates
                second_lead[0, :] = [n_contact[0], n_contact[1] - 4.3301]
                if n_contact[0] < lead[lead_index[1], 0] + 10:
                    second_lead[1, :] = [second_lead[0, 0] - 10, second_lead[0, 1]]
                    second_lead[2, :] = [second_lead[1, 0], lead[lead_index[1], 1]]
                    second_lead[3, :] = [lead[lead_index[1], 0] + 10, second_lead[2, 1]]
                    second_lead[4, :] = [second_lead[3, 0], second_lead[3, 1] - 10]
                    second_lead[5, :] = [second_lead[1, 0] - 10, second_lead[4, 1]]
                    second_lead[6, :] = [second_lead[5, 0], second_lead[0, 1] + 8.6602]
                else:
                    second_lead[1, :] = [second_lead[0, 0] - 10, second_lead[0, 1]]
                    second_lead[2, :] = [second_lead[1, 0], lead[lead_index[1], 1] - 10]
                    second_lead[3, :] = [lead[lead_index[1], 0] - 10, second_lead[2, 1]]
                    second_lead[4, :] = [second_lead[3, 0], second_lead[3, 1] + 10]
                    second_lead[5, :] = [second_lead[1, 0] - 10, second_lead[4, 1]]
                    second_lead[6, :] = [second_lead[5, 0], second_lead[0, 1] + 8.6602]
                second_lead[7, :] = [second_lead[0, 0], second_lead[6, 1]]
                second_lead[8, :] = second_lead[0, :]
        elif orientation_path == 'down':
            lead_index = np.array([1, 0])
            if angle < 0:  # Angle of the wire is negative [ \ ]
                first_lead[0, :] = [p_contact[0], p_contact[1] - 4.3301]
                if p_contact[0] < lead[lead_index[0], 0] + 10:
                    first_lead[1, :] = [first_lead[0, 0] - 10, first_lead[0, 1]]
                    first_lead[2, :] = [first_lead[1, 0], lead[lead_index[0], 1]]
                    first_lead[3, :] = [lead[lead_index[0], 0] + 10, first_lead[2, 1]]
                    first_lead[4, :] = [first_lead[3, 0], first_lead[3, 1] - 10]
                    first_lead[5, :] = [first_lead[1, 0] - 10, first_lead[4, 1]]
                    first_lead[6, :] = [first_lead[5, 0], first_lead[0, 1] + 8.6602]
                else:
                    first_lead[1, :] = [first_lead[0, 0] - 10, first_lead[0, 1]]
                    first_lead[2, :] = [first_lead[1, 0], lead[lead_index[0], 1] - 10]
                    first_lead[3, :] = [lead[lead_index[0], 0] - 10, first_lead[2, 1]]
                    first_lead[4, :] = [first_lead[3, 0], first_lead[3, 1] + 10]
                    first_lead[5, :] = [first_lead[1, 0] - 10, first_lead[4, 1]]
                    first_lead[6, :] = [first_lead[5, 0], first_lead[0, 1] + 8.6602]
                first_lead[7, :] = [first_lead[0, 0], first_lead[6, 1]]
                first_lead[8, :] = first_lead[0, :]
                # second_lead Coordinates
                second_lead[0, :] = [n_contact[0], n_contact[1] - 4.3301]
                if n_contact[0] < lead[lead_index[1], 0] - 10:
                    second_lead[1, :] = [second_lead[0, 0] + 20, second_lead[0, 1]]
                    second_lead[2, :] = [second_lead[1, 0], lead[lead_index[1], 1]]
                    second_lead[3, :] = [lead[lead_index[1], 0] + 10, second_lead[2, 1]]
                    second_lead[4, :] = [second_lead[3, 0], second_lead[3, 1] + 10]
                    second_lead[5, :] = [second_lead[1, 0] - 10, second_lead[4, 1]]
                    second_lead[6, :] = [second_lead[5, 0], second_lead[0, 1] + 8.6602]
                else:
                    second_lead[1, :] = [second_lead[0, 0] + 20, second_lead[0, 1]]
                    second_lead[2, :] = [second_lead[1, 0], lead[lead_index[1], 1] + 10]
                    second_lead[3, :] = [lead[lead_index[1], 0] - 10, second_lead[2, 1]]
                    second_lead[4, :] = [second_lead[3, 0], second_lead[3, 1] - 10]
                    second_lead[5, :] = [second_lead[1, 0] - 10, second_lead[4, 1]]
                    second_lead[6, :] = [second_lead[5, 0], second_lead[0, 1] + 8.6602]
                second_lead[7, :] = [second_lead[0, 0], second_lead[6, 1]]
                second_lead[8, :] = second_lead[0, :]
            else:  # angle of wire is positive[ / ]
                first_lead[0, :] = [p_contact[0], p_contact[1] - 4.3301]
                if p_contact[0] < lead[lead_index[0], 0] - 10:
                    first_lead[1, :] = [first_lead[0, 0] + 10, first_lead[0, 1]]
                    first_lead[2, :] = [first_lead[1, 0], lead[lead_index[0], 1] - 10]
                    first_lead[3, :] = [lead[lead_index[0], 0] + 10, first_lead[2, 1]]
                    first_lead[4, :] = [first_lead[3, 0], first_lead[3, 1] + 10]
                    first_lead[5, :] = [first_lead[1, 0] + 10, first_lead[4, 1]]
                    first_lead[6, :] = [first_lead[5, 0], first_lead[0, 1] + 8.6602]
                else:
                    first_lead[1, :] = [first_lead[0, 0] + 10, first_lead[0, 1]]
                    first_lead[2, :] = [first_lead[1, 0], lead[lead_index[0], 1]]
                    first_lead[3, :] = [lead[lead_index[0], 0] - 10, first_lead[2, 1]]
                    first_lead[4, :] = [first_lead[3, 0], first_lead[3, 1] - 10]
                    first_lead[5, :] = [first_lead[1, 0] + 10, first_lead[4, 1]]
                    first_lead[6, :] = [first_lead[5, 0], first_lead[0, 1] + 8.6602]
                first_lead[7, :] = [first_lead[0, 0], first_lead[6, 1]]
                first_lead[8, :] = first_lead[0, :]
                # second_lead Coordinates
                second_lead[0, :] = [n_contact[0], n_contact[1] - 4.3301]
                if n_contact[0] < lead[lead_index[1], 0] + 10:
                    second_lead[1, :] = [second_lead[0, 0] - 20, second_lead[0, 1]]
                    second_lead[2, :] = [second_lead[1, 0], lead[lead_index[1], 1] + 10]
                    second_lead[3, :] = [lead[lead_index[1], 0] + 10, second_lead[2, 1]]
                    second_lead[4, :] = [second_lead[3, 0], second_lead[3, 1] - 10]
                    second_lead[5, :] = [second_lead[1, 0] + 10, second_lead[4, 1]]
                    second_lead[6, :] = [second_lead[5, 0], second_lead[0, 1] + 8.6602]
                else:
                    second_lead[1, :] = [second_lead[0, 0] - 20, second_lead[0, 1]]
                    second_lead[2, :] = [second_lead[1, 0], lead[lead_index[1], 1]]
                    second_lead[3, :] = [lead[lead_index[1], 0] - 10, second_lead[2, 1]]
                    second_lead[4, :] = [second_lead[3, 0], second_lead[3, 1] + 10]
                    second_lead[5, :] = [second_lead[1, 0] + 10, second_lead[4, 1]]
                    second_lead[6, :] = [second_lead[5, 0], second_lead[0, 1] + 8.6602]
                second_lead[7, :] = [second_lead[0, 0], second_lead[6, 1]]
                second_lead[8, :] = second_lead[0, :]
    return first_lead, second_lead

def horizontal_contact(p_contact, n_contact, lead, angle, orientation_path):
    first_lead = np.zeros([9, 2])
    second_lead = np.zeros([9, 2])
    if p_contact[1] < n_contact[1]:
        if orientation_path == 'left':
            lead_index = np.array([0, 1])
            if angle < 0:
                first_lead[0, :] = np.array([p_contact[0] - 5, p_contact[1]])
                if p_contact[1] < lead[lead_index[1], 1] + 10:
                    first_lead[1, :] = [first_lead[0, 0] + 20, first_lead[0, 1]]
                    first_lead[2, :] = [first_lead[1, 0], lead[lead_index[0], 1]]
                    first_lead[3, :] = [lead[lead_index[0], 0] + 10, first_lead[2, 1]]
                    first_lead[4, :] = [first_lead[3, 0], first_lead[3, 1] + 10]
                    first_lead[5, :] = [first_lead[1, 0] - 10, first_lead[4, 1]]
                    first_lead[6, :] = [first_lead[5, 0], first_lead[0, 1] + 8.6602]
                else:
                    first_lead[1, :] = [first_lead[0, 0], first_lead[0, 1] - 10]
                    first_lead[1, :] = [lead[lead_index[0], 0], first_lead[1, 1]]
                    first_lead[3, :] = [first_lead[2, 0], lead[lead_index[0], 1] - 10]
                    first_lead[4, :] = [first_lead[3, 0] + 10, first_lead[3, 1]]
                    first_lead[5, :] = [first_lead[4, 0], first_lead[1, 1] - 10]
                    first_lead[6, :] = [first_lead[0, 0] + 10, first_lead[5, 1]]
                first_lead[7, :] = [first_lead[0, 0], first_lead[6, 1]]
                first_lead[8, :] = first_lead[0, :]

                second_lead[0, :] = [n_contact[0] - 5, n_contact[1]]
                if n_contact[1] < lead[lead_index[1], 1] - 10:
                    second_lead[1, :] = [second_lead[0, 0], second_lead[0, 1] + 20]
                    second_lead[2, :] = [lead[lead_index[1], 0] - 10, second_lead[1, 1]]
                    second_lead[3, :] = [second_lead[2, 0], lead[lead_index[1], 1] + 10]
                    second_lead[4, :] = [second_lead[3, 0] + 10, second_lead[3, 1]]
                    second_lead[5, :] = [second_lead[4, 0], second_lead[1, 1] - 10]
                    second_lead[6, :] = [second_lead[1, 0] + 10, second_lead[5, 1]]
                else:
                    second_lead[1, :] = [second_lead[0, 0], second_lead[0, 1] + 20]
                    second_lead[2, :] = [lead[lead_index[1], 0], second_lead[1, 1]]
                    second_lead[3, :] = [second_lead[2, 0], lead[lead_index[1], 1] - 10]
                    second_lead[4, :] = [second_lead[3, 0] - 10, second_lead[3, 1]]
                    second_lead[5, :] = [second_lead[4, 0], second_lead[1, 1] - 10]
                    second_lead[6, :] = [second_lead[1, 0] + 10, second_lead[5, 1]]
                second_lead[7, :] = [second_lead[6, 0], second_lead[0, 1]]
                second_lead[8, :] = second_lead[0, :]

            else:  # angle of wire is positive [ / ]
                first_lead[0, :] = [p_contact[0] - 5, p_contact[1]]
                if p_contact[1] < lead[lead_index[0], 1] + 10:
                    first_lead[1, :] = [first_lead[0, 0], first_lead[0, 1] - 10]
                    first_lead[2, :] = [lead[lead_index[0], 0], first_lead[1, 1]]
                    first_lead[3, :] = [first_lead[2, 0], lead[lead_index[0], 1] + 10]
                    first_lead[4, :] = [first_lead[3, 0] - 10, first_lead[3, 1]]
                    first_lead[5, :] = [first_lead[4, 0], first_lead[1, 1] - 10]
                    first_lead[6, :] = [first_lead[0, 0] + 10, first_lead[5, 1]]
                else:
                    first_lead[1, :] = [first_lead[0, 0], first_lead[0, 1] - 10]
                    first_lead[2, :] = [lead[lead_index[0], 0] - 10, first_lead[1, 1]]
                    first_lead[3, :] = [first_lead[2, 0], lead[lead_index[0], 1] - 10]
                    first_lead[4, :] = [first_lead[3, 0] + 10, first_lead[3, 1]]
                    first_lead[5, :] = [first_lead[4, 0], first_lead[1, 1] - 10]
                    first_lead[6, :] = [first_lead[0, 0] + 10, first_lead[5, 1]]
                first_lead[7, :] = [first_lead[6, 0], first_lead[0, 1]]
                first_lead[8, :] = first_lead[0, :]

                # second_leadCoordinates
                second_lead[0, :] = [n_contact[0] - 5, n_contact[1]]
                if p_contact[1] < lead[lead_index[1], 1] - 10:
                    second_lead[1, :] = [second_lead[0, 0], second_lead[0, 1] + 20]
                    second_lead[2, :] = [lead[lead_index[1], 0], second_lead[1, 1]]
                    second_lead[3, :] = [second_lead[2, 0], lead[lead_index[1], 1] + 10]
                    second_lead[4, :] = [second_lead[3, 0] + 10, second_lead[3, 1]]
                    second_lead[5, :] = [second_lead[4, 0], second_lead[1, 1] - 10]
                    second_lead[6, :] = [second_lead[1, 0] + 10, second_lead[5, 1]]
                else:
                    second_lead[1, :] = [second_lead[0, 0], second_lead[0, 1] + 20]
                    second_lead[2, :] = [lead[lead_index[1], 0] + 10, second_lead[1, 1]]
                    second_lead[3, :] = [second_lead[2, 0], lead[lead_index[1], 1] - 10]
                    second_lead[4, :] = [second_lead[3, 0] - 10, second_lead[3, 1]]
                    second_lead[5, :] = [second_lead[4, 0], second_lead[1, 1] - 10]
                    second_lead[6, :] = [second_lead[1, 0] + 10, second_lead[5, 1]]
                second_lead[7, :] = [second_lead[6, 0], second_lead[0, 1]]
                second_lead[8, :] = second_lead[0, :]
        elif orientation_path == 'right':
            lead_index = np.array([1, 0])
            if angle < 0:  # Angle of the wire is negative [ \ ]
                # first_leadCoordinates
                first_lead[0, :] = [p_contact[0] - 5, p_contact[1]]
                if p_contact[1] < lead[lead_index[0], 1] + 10:
                    first_lead[1, :] = [first_lead[0, 0], first_lead[0, 1] - 20]
                    first_lead[2, :] = [lead[lead_index[0], 0] + 10, first_lead[1, 1]]
                    first_lead[3, :] = [first_lead[2, 0], lead[lead_index[0], 1] + 10]
                    first_lead[4, :] = [first_lead[3, 0] - 10, first_lead[3, 1]]
                    first_lead[5, :] = [first_lead[4, 0], first_lead[1, 1] + 10]
                    first_lead[6, :] = [first_lead[0, 0] + 10, first_lead[5, 1]]
                else:
                    first_lead[1, :] = [first_lead[0, 0], first_lead[0, 1] - 20]
                    first_lead[2, :] = [lead[lead_index[0], 0], first_lead[1, 1]]
                    first_lead[3, :] = [first_lead[2, 0], lead[lead_index[0], 1] - 10]
                    first_lead[4, :] = [first_lead[3, 0] + 10, first_lead[3, 1]]
                    first_lead[5, :] = [first_lead[4, 0], first_lead[1, 1] + 10]
                    first_lead[6, :] = [first_lead[0, 0] + 10, first_lead[5, 1]]
                first_lead[7, :] = [first_lead[6, 0], first_lead[0, 1]]
                first_lead[8, :] = first_lead[0, :]
                # second_leadCoordinates
                second_lead[0, :] = [n_contact[0] - 5, n_contact[1]]
                if n_contact[1] < lead[lead_index[1], 1] - 10:
                    second_lead[1, :] = [second_lead[0, 0], second_lead[0, 1] + 10]
                    second_lead[2, :] = [lead[lead_index[1], 0] - 10, second_lead[1, 1]]
                    second_lead[3, :] = [second_lead[2, 0], lead[lead_index[1], 1] + 10]
                    second_lead[4, :] = [second_lead[3, 0] + 10, second_lead[3, 1]]
                    second_lead[5, :] = [second_lead[4, 0], second_lead[1, 1] + 10]
                    second_lead[6, :] = [second_lead[1, 0] + 10, second_lead[5, 1]]
                else:
                    second_lead[1, :] = [second_lead[0, 0], second_lead[0, 1] + 10]
                    second_lead[2, :] = [lead[lead_index[1], 0], second_lead[1, 1]]
                    second_lead[3, :] = [second_lead[2, 0], lead[lead_index[1], 1] - 10]
                    second_lead[4, :] = [second_lead[3, 0] - 10, second_lead[3, 1]]
                    second_lead[5, :] = [second_lead[4, 0], second_lead[1, 1] + 10]
                    second_lead[6, :] = [second_lead[1, 0] + 10, second_lead[5, 1]]
                second_lead[7, :] = [second_lead[6, 0], second_lead[0, 1]]
                second_lead[8, :] = second_lead[0, :]
            else:  # angle of wire is positive [ / ]
                first_lead[0, :] = [p_contact[0] - 5, p_contact[1]]
                if p_contact[1] < lead[lead_index[0], 1] + 10:
                    first_lead[1, :] = [first_lead[0, 0], first_lead[0, 1] - 20]
                    first_lead[2, :] = [lead[lead_index[0], 0] + 10, first_lead[1, 1]]
                    first_lead[3, :] = [first_lead[2, 0], lead[lead_index[0], 1] + 10]
                    first_lead[4, :] = [first_lead[3, 0] - 10, first_lead[3, 1]]
                    first_lead[5, :] = [first_lead[4, 0], first_lead[1, 1] + 10]
                    first_lead[6, :] = [first_lead[0, 0] + 10, first_lead[5, 1]]
                else:
                    first_lead[1, :] = [first_lead[0, 0], first_lead[0, 1] - 20]
                    first_lead[2, :] = [lead[lead_index[0], 0], first_lead[1, 1]]
                    first_lead[3, :] = [first_lead[2, 0], lead[lead_index[0], 1] - 10]
                    first_lead[4, :] = [first_lead[3, 0] + 10, first_lead[3, 1]]
                    first_lead[5, :] = [first_lead[4, 0], first_lead[1, 1] + 10]
                    first_lead[6, :] = [first_lead[0, 0] + 10, first_lead[5, 1]]
                first_lead[7, :] = [first_lead[6, 0], first_lead[0, 1]]
                first_lead[8, :] = first_lead[0, :]
                # second_leadCoordinates
                second_lead[0, :] = [n_contact[0] - 5, n_contact[1]]
                if n_contact[1] < lead[lead_index[1], 1] - 10:
                    second_lead[1, :] = [second_lead[0, 0], second_lead[0, 1] + 10]
                    second_lead[2, :] = [lead[lead_index[1], 0] - 10, second_lead[1, 1]]
                    second_lead[3, :] = [second_lead[2, 0], lead[lead_index[1], 1] + 10]
                    second_lead[4, :] = [second_lead[3, 0] + 10, second_lead[3, 1]]
                    second_lead[5, :] = [second_lead[4, 0], second_lead[1, 1] + 10]
                    second_lead[6, :] = [second_lead[1, 0] + 10, second_lead[5, 1]]
                else:
                    second_lead[1, :] = [second_lead[0, 0], second_lead[0, 1] + 10]
                    second_lead[2, :] = [lead[lead_index[1], 0], second_lead[1, 1]]
                    second_lead[3, :] = [second_lead[2, 0], lead[lead_index[1], 1] - 10]
                    second_lead[4, :] = [second_lead[3, 0] - 10, second_lead[3, 1]]
                    second_lead[5, :] = [second_lead[4, 0], second_lead[1, 1] + 10]
                    second_lead[6, :] = [second_lead[1, 0] + 10, second_lead[5, 1]]
                second_lead[7, :] = [second_lead[6, 0], second_lead[0, 1]]
                second_lead[8, :] = second_lead[0, :]
    else:  # p extremity is up
        if orientation_path == "left":
            lead_index = np.array([0, 1])
            if angle < 0:  # Angle of the wire is negative [ \ ]
                first_lead[0, :] = [p_contact[0] - 5, p_contact[1]]
                if p_contact[1] < lead[lead_index[0], 1] - 10:
                    first_lead[1, :] = [first_lead[0, 0], first_lead[0, 1] + 10]
                    first_lead[2, :] = [lead[lead_index[0], 0], first_lead[1, 1]]
                    first_lead[3, :] = [first_lead[2, 0], lead[lead_index[0], 1] + 10]
                    first_lead[4, :] = [first_lead[3, 0] + 10, first_lead[3, 1]]
                    first_lead[5, :] = [first_lead[4, 0], first_lead[1, 1] + 10]
                    first_lead[6, :] = [first_lead[0, 0] + 10, first_lead[5, 1]]
                else:
                    first_lead[1, :] = [first_lead[0, 0], first_lead[0, 1] + 10]
                    first_lead[2, :] = [lead[lead_index[0], 0], first_lead[1, 1]]
                    first_lead[3, :] = [first_lead[2, 0], lead[lead_index[0], 1] - 10]
                    first_lead[4, :] = [first_lead[3, 0] - 10, first_lead[3, 1]]
                    first_lead[5, :] = [first_lead[4, 0], first_lead[1, 1] + 10]
                    first_lead[6, :] = [first_lead[0, 0] + 10, first_lead[5, 1]]
                first_lead[7, :] = [first_lead[6, 0], first_lead[0, 1]]
                first_lead[8, :] = first_lead[0, :]
                # second_leadCoordinates
                second_lead[0, :] = [n_contact[0] - 5, n_contact[1]]
                if n_contact[1] < lead[lead_index[1], 1] + 10:
                    second_lead[1, :] = [second_lead[0, 0], second_lead[0, 1] - 20]
                    second_lead[2, :] = [lead[lead_index[1], 0] + 10, second_lead[1, 1]]
                    second_lead[3, :] = [second_lead[2, 0], lead[lead_index[1], 1] + 10]
                    second_lead[4, :] = [second_lead[3, 0] - 10, second_lead[3, 1]]
                    second_lead[5, :] = [second_lead[4, 0], second_lead[1, 1] + 10]
                    second_lead[6, :] = [second_lead[1, 0] + 10, second_lead[5, 1]]
                else:
                    second_lead[1, :] = [second_lead[0, 0], second_lead[0, 1] - 20]
                    second_lead[2, :] = [lead[lead_index[1], 0], second_lead[1, 1]]
                    second_lead[3, :] = [second_lead[2, 0], lead[lead_index[1], 1] - 10]
                    second_lead[4, :] = [second_lead[3, 0] + 10, second_lead[3, 1]]
                    second_lead[5, :] = [second_lead[4, 0], second_lead[1, 1] + 10]
                    second_lead[6, :] = [second_lead[1, 0] + 10, second_lead[5, 1]]
                second_lead[7, :] = [second_lead[6, 0], second_lead[0, 1]]
                second_lead[8, :] = second_lead[0, :]
            else:  # angle of wire is positive [ / ]
                first_lead[0, :] = [p_contact[0] - 5, p_contact[1]]
                if p_contact[1] < lead[lead_index[0], 1] - 10:
                    first_lead[1, :] = [first_lead[0, 0], first_lead[0, 1] + 10]
                    first_lead[2, :] = [lead[lead_index[0], 0], first_lead[1, 1]]
                    first_lead[3, :] = [first_lead[2, 0], lead[lead_index[0], 1] + 10]
                    first_lead[4, :] = [first_lead[3, 0] + 10, first_lead[3, 1]]
                    first_lead[5, :] = [first_lead[4, 0], first_lead[1, 1] + 10]
                    first_lead[6, :] = [first_lead[0, 0] + 10, first_lead[5, 1]]
                else:
                    first_lead[1, :] = [first_lead[0, 0], first_lead[0, 1] + 10]
                    first_lead[2, :] = [lead[lead_index[0], 0], first_lead[1, 1]]
                    first_lead[3, :] = [first_lead[2, 0], lead[lead_index[0], 1] - 10]
                    first_lead[4, :] = [first_lead[3, 0] - 10, first_lead[3, 1]]
                    first_lead[5, :] = [first_lead[4, 0], first_lead[1, 1] + 10]
                    first_lead[6, :] = [first_lead[0, 0] + 10, first_lead[5, 1]]
                first_lead[7, :] = [first_lead[6, 0], first_lead[0, 1]]
                first_lead[8, :] = first_lead[0, :]
                # second_leadCoordinates
                second_lead[0, :] = [n_contact[0] - 5, n_contact[1]]
                if n_contact[1] < lead[lead_index[1], 1] + 10:
                    second_lead[1, :] = [second_lead[0, 0], second_lead[0, 1] - 20]
                    second_lead[2, :] = [lead[lead_index[1], 0] + 10, second_lead[1, 1]]
                    second_lead[3, :] = [second_lead[2, 0], lead[lead_index[1], 1] + 10]
                    second_lead[4, :] = [second_lead[3, 0] - 10, second_lead[3, 1]]
                    second_lead[5, :] = [second_lead[4, 0], second_lead[1, 1] + 10]
                    second_lead[6, :] = [second_lead[1, 0] + 10, second_lead[5, 1]]
                else:
                    second_lead[1, :] = [second_lead[0, 0], second_lead[0, 1] - 20]
                    second_lead[2, :] = [lead[lead_index[1], 0], second_lead[1, 1]]
                    second_lead[3, :] = [second_lead[2, 0], lead[lead_index[1], 1] - 10]
                    second_lead[4, :] = [second_lead[3, 0] + 10, second_lead[3, 1]]
                    second_lead[5, :] = [second_lead[4, 0], second_lead[1, 1] + 10]
                    second_lead[6, :] = [second_lead[1, 0] + 10, second_lead[5, 1]]
                second_lead[7, :] = [second_lead[6, 0], second_lead[0, 1]]
                second_lead[8, :] = second_lead[0, :]
        elif orientation_path == 'right':
            lead_index = np.array([1, 0])
            if angle < 0:  # Angle of the wire is negative [ \ ]
                first_lead[0, :] = [p_contact[0] - 5, p_contact[1]]
                if p_contact[1] < lead[lead_index[0], 1] - 10:
                    first_lead[1, :] = [first_lead[0, 0], first_lead[0, 1] + 20]
                    first_lead[2, :] = [lead[lead_index[0], 0], first_lead[1, 1]]
                    first_lead[3, :] = [first_lead[2, 0], lead[lead_index[0], 1] + 10]
                    first_lead[4, :] = [first_lead[3, 0] + 10, first_lead[3, 1]]
                    first_lead[5, :] = [first_lead[4, 0], first_lead[1, 1] - 10]
                    first_lead[6, :] = [first_lead[0, 0] + 10, first_lead[5, 1]]
                else:
                    first_lead[1, :] = [first_lead[0, 0], first_lead[0, 1] + 20]
                    first_lead[2, :] = [lead[lead_index[0], 0] + 10, first_lead[1, 1]]
                    first_lead[3, :] = [first_lead[2, 0], lead[lead_index[0], 1] - 10]
                    first_lead[4, :] = [first_lead[3, 0] - 10, first_lead[3, 1]]
                    first_lead[5, :] = [first_lead[4, 0], first_lead[1, 1] - 10]
                    first_lead[6, :] = [first_lead[0, 0] + 10, first_lead[5, 1]]
                first_lead[7, :] = [first_lead[6, 0], first_lead[0, 1]]
                first_lead[8, :] = first_lead[0, :]
                # second_leadCoordinates
                second_lead[0, :] = [n_contact[0] - 5, n_contact[1]]
                if n_contact[1] < lead[lead_index[1], 1] + 10:
                    second_lead[1, :] = [second_lead[0, 0], second_lead[0, 1] - 10]
                    second_lead[2, :] = [lead[lead_index[1], 0], second_lead[1, 1]]
                    second_lead[3, :] = [second_lead[2, 0], lead[lead_index[1], 1] + 10]
                    second_lead[4, :] = [second_lead[3, 0] - 10, second_lead[3, 1]]
                    second_lead[5, :] = [second_lead[4, 0], second_lead[1, 1] - 10]
                    second_lead[6, :] = [second_lead[1, 0] + 10, second_lead[5, 1]]
                else:
                    second_lead[1, :] = [second_lead[0, 0], second_lead[0, 1] - 10]
                    second_lead[2, :] = [lead[lead_index[1], 0] - 10, second_lead[1, 1]]
                    second_lead[3, :] = [second_lead[2, 0], lead[lead_index[1], 1] - 10]
                    second_lead[4, :] = [second_lead[3, 0] + 10, second_lead[3, 1]]
                    second_lead[5, :] = [second_lead[4, 0], second_lead[1, 1] - 10]
                    second_lead[6, :] = [second_lead[1, 0] + 10, second_lead[5, 1]]
                second_lead[7, :] = [second_lead[6, 0], second_lead[0, 1]]
                second_lead[8, :] = second_lead[0, :]
            else:  # angle of wire is positive [ / ]
                first_lead[0, :] = [p_contact[0] - 5, p_contact[1]]
                if p_contact[1] < lead[lead_index[0], 1] - 10:
                    first_lead[1, :] = [first_lead[0, 0], first_lead[0, 1] + 20]
                    first_lead[2, :] = [lead[lead_index[0], 0], first_lead[1, 1]]
                    first_lead[3, :] = [first_lead[2, 0], lead[lead_index[0], 1] + 10]
                    first_lead[4, :] = [first_lead[3, 0] + 10, first_lead[3, 1]]
                    first_lead[5, :] = [first_lead[4, 0], first_lead[1, 1] - 10]
                    first_lead[6, :] = [first_lead[0, 0] + 10, first_lead[5, 1]]
                else:
                    first_lead[1, :] = [first_lead[0, 0], first_lead[0, 1] + 20]
                    first_lead[2, :] = [lead[lead_index[0], 0] + 10, first_lead[1, 1]]
                    first_lead[3, :] = [first_lead[2, 0], lead[lead_index[0], 1] - 10]
                    first_lead[4, :] = [first_lead[3, 0] - 10, first_lead[3, 1]]
                    first_lead[5, :] = [first_lead[4, 0], first_lead[1, 1] - 10]
                    first_lead[6, :] = [first_lead[0, 0] + 10, first_lead[5, 1]]
                first_lead[7, :] = [first_lead[6, 0], first_lead[0, 1]]
                first_lead[8, :] = first_lead[0, :]
                # second_leadCoordinates
                second_lead[0, :] = [n_contact[0] - 5, n_contact[1]]
                if n_contact[1] < lead[lead_index[1], 1] + 10:
                    second_lead[1, :] = [second_lead[0, 0], second_lead[0, 1] - 10]
                    second_lead[2, :] = [lead[lead_index[1], 0], second_lead[1, 1]]
                    second_lead[3, :] = [second_lead[2, 0], lead[lead_index[1], 1] + 10]
                    second_lead[4, :] = [second_lead[3, 0] - 10, second_lead[3, 1]]
                    second_lead[5, :] = [second_lead[4, 0], second_lead[1, 1] - 10]
                    second_lead[6, :] = [second_lead[1, 0] + 10, second_lead[5, 1]]
                else:
                    second_lead[1, :] = [second_lead[0, 0], second_lead[0, 1] - 10]
                    second_lead[2, :] = [lead[lead_index[1], 0] - 10, second_lead[1, 1]]
                    second_lead[3, :] = [second_lead[2, 0], lead[lead_index[1], 1] - 10]
                    second_lead[4, :] = [second_lead[3, 0] + 10, second_lead[3, 1]]
                    second_lead[5, :] = [second_lead[4, 0], second_lead[1, 1] - 10]
                    second_lead[6, :] = [second_lead[1, 0] + 10, second_lead[5, 1]]
                second_lead[7, :] = [second_lead[6, 0], second_lead[0, 1]]
                second_lead[8, :] = second_lead[0, :]
    return first_lead, second_lead