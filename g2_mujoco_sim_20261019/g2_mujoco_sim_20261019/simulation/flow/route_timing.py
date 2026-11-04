"""Read physical segment durations while retaining legacy OMPL playback timing."""
import numpy as np


def segment_durations(record,vehicle):
    poses = record['poses']
    if 'timestamps' in record:
        times = np.asarray(record['timestamps'],dtype=float)
        if times.shape!=(len(poses),) or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
            raise ValueError('Invalid physical route timestamps')
        return np.diff(times)
    forward = record['longitudinal']>=0
    speeds = np.where(forward,vehicle['speed_m_s'],vehicle['reverse_speed_m_s'])
    durations = np.maximum(record['ds']/speeds,np.abs(record['dyaw'])/vehicle['max_yaw_speed_rad_s'])
    switches = np.r_[False,forward[1:]!=forward[:-1]]
    return durations+switches*vehicle['switch_gear_hold_s']
