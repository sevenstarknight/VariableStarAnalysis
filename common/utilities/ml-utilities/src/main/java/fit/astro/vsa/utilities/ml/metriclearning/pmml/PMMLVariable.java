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
package fit.astro.vsa.utilities.ml.metriclearning.pmml;

import org.apache.commons.math3.linear.RealMatrix;

/**
 *
 * @author Kyle Johnston
 */
public class PMMLVariable {

    private RealMatrix mk;

    private final double tau;
    private final double mu;

    /**
     * 
     * @param lk
     * @param tau
     * @param mu 
     */
    public PMMLVariable(RealMatrix lk,  double tau, double mu) {
        this.mk = lk;
        this.tau = tau;
        this.mu = mu;
    }

    public RealMatrix getMk() {
        return mk;
    }

    public void setMk(RealMatrix mk) {
        this.mk = mk;
    }

    public double getTau() {
        return tau;
    }

    public double getMu() {
        return mu;
    }

}
