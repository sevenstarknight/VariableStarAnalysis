/*
 * Copyright (C) 2018 Kyle Johnston 
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package fit.astro.vsa.common.utilities.clustering.data;

import java.util.List;
import java.util.Map;
import org.apache.commons.math3.linear.RealMatrix;

/**
 *
 * @author Kyle Johnston 
 */
public class ClusterOutput {

    private final Map<Integer, RealMatrix> clusterCenters;
    private final Map<Integer, List<Integer>> clusterMembers;

    public ClusterOutput(Map<Integer, RealMatrix> clusterCenters, 
            Map<Integer, List<Integer>> clusterMembers) {
        this.clusterCenters = clusterCenters;
        this.clusterMembers = clusterMembers;
    }

    public Map<Integer, RealMatrix> getClusterCenters() {
        return clusterCenters;
    }

    public Map<Integer, List<Integer>> getClusterMembers() {
        return clusterMembers;
    }

}
