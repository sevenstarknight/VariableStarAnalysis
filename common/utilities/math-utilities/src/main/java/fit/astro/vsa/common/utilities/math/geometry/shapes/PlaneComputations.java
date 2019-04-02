/*
 * Copyright (C) 2016 Kyle Johnston
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without isEven the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package fit.astro.vsa.common.utilities.math.geometry.shapes;

import fit.astro.vsa.common.bindings.math.geometry.LineEquation;
import fit.astro.vsa.common.bindings.math.geometry.PlaneEquation;
import fit.astro.vsa.common.utilities.math.NumericTests;
import fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;

/**
 * Parametric Equation of Intersecting Line in XYZ coordinates between planes
 * generated http://mathworld.wolfram.com/Plane-PlaneIntersection.html
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class PlaneComputations {

    // Empty Constructor
    private PlaneComputations() {
    }

    /**
     * ========================================================== Determine
     * Parametric Equation of Intersecting Line in XYZ coordinates between
     * planes generated
     * http://mathworld.wolfram.com/Plane-PlaneIntersection.html
     * <p>
     * @param planeEquation1
     * @param planeEquation2
     * <p>
     * @return
     */
    public static LineEquation generatePlanePlaneIntersection(
            PlaneEquation planeEquation1,
            PlaneEquation planeEquation2) {

        // Make sure the two are not parallel
        if (NumericTests.isApproxEqual(planeEquation1.getaCoeff() / planeEquation2.getaCoeff(), planeEquation1.getbCoeff() / planeEquation2.getbCoeff())
                && NumericTests.isApproxEqual(planeEquation1.getaCoeff() / planeEquation2.getaCoeff(), planeEquation1.getcCoeff() / planeEquation2.getcCoeff())) {
            throw new ArithmeticException("Planes are Parallel, "
                    + "Can't Find Intersection");
        }

        //==========================================================
        // Determine Parametric Equation of Intersecting Line in
        // XYZ coordinates between planes generated
        // http://mathworld.wolfram.com/Plane-PlaneIntersection.html
        double xEst, yEst, zEst;
        if (!NumericTests.isApproxZero(planeEquation2.getcCoeff())
                && !NumericTests.isApproxZero(planeEquation1.getcCoeff())) {
            zEst = 0.0;
            yEst = (planeEquation2.getaCoeff() - planeEquation1.getaCoeff())
                    / (planeEquation1.getbCoeff() * planeEquation2.getaCoeff()
                    - planeEquation2.getbCoeff() * planeEquation1.getaCoeff());

            xEst = (planeEquation2.getbCoeff() - planeEquation1.getbCoeff())
                    / (planeEquation1.getaCoeff() * planeEquation2.getbCoeff()
                    - planeEquation2.getaCoeff() * planeEquation1.getbCoeff());

        } else {
            yEst = 0.0;
            zEst = (planeEquation2.getaCoeff() - planeEquation1.getaCoeff())
                    / (planeEquation1.getcCoeff() * planeEquation2.getaCoeff()
                    - planeEquation2.getcCoeff() * planeEquation1.getaCoeff());

            xEst = (planeEquation2.getcCoeff() - planeEquation1.getcCoeff())
                    / (planeEquation1.getaCoeff() * planeEquation2.getcCoeff()
                    - planeEquation2.getaCoeff() * planeEquation1.getcCoeff());
        }

        //==========================================================
        // x_0 and crossResults (a) define the equation of the line of
        // intersection between the two planes via x_0 + t*a, where t
        // is a parameter, line is in XYZ coordinates
        RealVector x0Plane = MatrixUtils.createRealVector(
                new double[]{xEst, yEst, zEst});
        RealVector crossResults = MatrixUtils.createRealVector(VectorOperations.cross3(
                planeEquation1.getCoefficients().toArray(),
                planeEquation2.getCoefficients().toArray()));

        return new LineEquation(x0Plane, crossResults);

    }

}
