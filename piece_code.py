#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 20:07:56 2019

@author: rcuello
"""

        
        dt = np.array([data[i][2] for i in listSelected()])
        sel_dist = np.array(dt)
        if sk_id>0:
            sub_data = []
            skeleton = sk_list[sk_id-1]
            for kp_a,kp_b in sub_list:
                p1 = skeleton[kp_a]
                p2 = skeleton[kp_b]
                sk_feat = calc_descriptor_feats(p1,p2)
                sub_data.append(sk_feat)
            last_dist = np.array([[sub_data[i][2] for i in range(38)]])
        diff_dist = sel_dist - last_dist
        s1 = scale_linear_bycolumn(feat_diff).flatten().astype(np.uint8)
        s2 = scale_linear_bycolumn(diff_dist.reshape(38,1)).flatten().astype(np.uint8)
        s3 = scale_linear_bycolumn(data).flatten().astype(np.uint8)
        # Now, merging all data
        z = np.append(s1,s2)
        z = np.append(z,s3)
        return z