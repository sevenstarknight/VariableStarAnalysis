/*
 * Copyright (C) 2016 Kyle Johnston
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
package fit.astro.vsa.common.utilities.clustering.emgmm;

import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class ClusterGM {

    private final int id;
    private double prior;

    private RealVector center;
    private RealMatrix cov;

    private RealMatrix invCov;
    private double detCov;

    /**
     *
     * @param id
     * @param center
     * @param cov
     * @param clusters
     */
    public ClusterGM(int id, RealVector center, RealMatrix cov, int clusters) {
        this.id = id;
        this.center = center;
        this.cov = cov;

        LUDecomposition decomposition = new LUDecomposition(cov);

        this.invCov = MatrixUtils.inverse(cov);
        this.detCov = 1.0 / Math.sqrt(decomposition.getDeterminant());

        this.prior = 1.0 / (double) clusters;

    }

    public void updateCluster(RealVector center, RealMatrix cov, double prior) {
        this.center = center;
        this.cov = cov;

        LUDecomposition decomposition = new LUDecomposition(cov);

        this.invCov = MatrixUtils.inverse(cov);
        this.detCov = 1.0 / Math.sqrt(decomposition.getDeterminant());
        
        this.prior = prior;
    }

    public int getId() {
        return id;
    }

    /**
     * @return the center
     */
    public RealVector getCenter() {
        return center;
    }

    /**
     * @return the cov
     */
    public RealMatrix getCov() {
        return cov;
    }

    /**
     * @return the invCov
     */
    public RealMatrix getInvCov() {
        return invCov;
    }

    /**
     * @return the detCov
     */
    public double getDetCov() {
        return detCov;
    }

    /**
     * @return the prior
     */
    public double getPrior() {
        return prior;
    }

}
