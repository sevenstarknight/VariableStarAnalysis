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
package fit.astro.vsa.analysis.feature.smoothing.util;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle.Johnston
 */
public class SuperSmootherResults {

    private RealVector smo_n;
    private RealVector resid;

    /**
     *
     * @param smo_n
     * @param resid
     */
    public SuperSmootherResults(RealVector smo_n, RealVector resid) {
        this.resid = resid;
        this.smo_n = smo_n;
    }

    /**
     *
     * @param smo_n
     * @param resid
     */
    public SuperSmootherResults(double[] smo_n, double[] resid) {
        this.resid = MatrixUtils.createRealVector(resid);
        this.smo_n = MatrixUtils.createRealVector(smo_n);
    }

    /**
     * @return the smo_n
     */
    public RealVector getSmo_n() {
        return smo_n;
    }

    /**
     * @param smo_n the smo_n to set
     */
    public void setSmo_n(RealVector smo_n) {
        this.smo_n = smo_n;
    }

    /**
     * @return the resid
     */
    public RealVector getResid() {
        return resid;
    }

    /**
     * @param resid the resid to set
     */
    public void setResid(RealVector resid) {
        this.resid = resid;
    }

}
