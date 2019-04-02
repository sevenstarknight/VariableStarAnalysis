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
package fit.astro.vsa.utilities.ml.metriclearning.l3ml_mv;

import org.apache.commons.math3.linear.RealMatrix;

/**
 *
 * @author Kyle Johnston
 */
public class L3MLVariable_MV {

    private RealMatrix gammak;
    private RealMatrix nuk;
    private double weight;

    /**
     *
     * @param gammak
     * @param nuk
     * @param weight
     */
    public L3MLVariable_MV(RealMatrix gammak, RealMatrix nuk,
            double weight) {
        this.gammak = gammak;
        this.nuk = nuk;
        this.weight = weight;
    }

    public RealMatrix getUk() {
        return (gammak.transpose()).multiply(gammak);
    }

    public RealMatrix getVk() {
        return (nuk.transpose()).multiply(nuk);
    }

    public RealMatrix getGammak() {
        return gammak;
    }

    public void setGammak(RealMatrix gammak) {
        this.gammak = gammak;
    }

    public RealMatrix getNuk() {
        return nuk;
    }

    public void setNuk(RealMatrix nuk) {
        this.nuk = nuk;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

}
