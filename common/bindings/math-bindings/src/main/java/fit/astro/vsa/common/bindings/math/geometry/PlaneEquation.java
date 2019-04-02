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
package fit.astro.vsa.common.bindings.math.geometry;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class PlaneEquation {

    private final RealVector coefficients;
    private final double aCoeff;
    private final double bCoeff;
    private final double cCoeff;

    /**
     *
     * @param coefficients
     */
    public PlaneEquation(RealVector coefficients) {
        this.coefficients = coefficients;
        this.aCoeff = coefficients.getEntry(0);
        this.bCoeff = coefficients.getEntry(1);
        this.cCoeff = coefficients.getEntry(2);
    }
    
    /**
     * 
     * @param a
     * @param b
     * @param c 
     */
    public PlaneEquation(double a, double b, double c) {
        this.coefficients = MatrixUtils.createRealVector(
                new double[]{a,b,c});
        this.aCoeff = a;
        this.bCoeff = b;
        this.cCoeff = c;
    }

    /**
     * @return the coefficients
     */
    public RealVector getCoefficients() {
        return coefficients;
    }

    /**
     * @return the aCoeff
     */
    public double getaCoeff() {
        return aCoeff;
    }

    /**
     * @return the bCoeff
     */
    public double getbCoeff() {
        return bCoeff;
    }

    /**
     * @return the cCoeff
     */
    public double getcCoeff() {
        return cCoeff;
    }

}
