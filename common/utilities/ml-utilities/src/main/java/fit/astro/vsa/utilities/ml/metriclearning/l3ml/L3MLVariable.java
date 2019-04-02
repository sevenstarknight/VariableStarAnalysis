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
package fit.astro.vsa.utilities.ml.metriclearning.l3ml;

import org.apache.commons.math3.linear.RealMatrix;

/**
 *
 * @author Kyle Johnston
 */
public class L3MLVariable {

    private RealMatrix lk;
    private double weight;

    private final double tau;
    private final double mu;

    /**
     * 
     * @param lk
     * @param weight
     * @param tau
     * @param mu 
     */
    public L3MLVariable(RealMatrix lk, double weight, double tau, double mu) {
        this.lk = lk;
        this.weight = weight;
        this.tau = tau;
        this.mu = mu;
    }

    public RealMatrix getLk() {
        return lk;
    }

    public void setLk(RealMatrix lk) {
        this.lk = lk;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public double getTau() {
        return tau;
    }

    public double getMu() {
        return mu;
    }

}
