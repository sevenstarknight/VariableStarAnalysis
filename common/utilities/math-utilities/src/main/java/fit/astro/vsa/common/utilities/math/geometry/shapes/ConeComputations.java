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
package fit.astro.vsa.common.utilities.math.geometry.shapes;

import fit.astro.vsa.common.bindings.math.geometry.Angle;
import fit.astro.vsa.common.bindings.math.geometry.ConeEquation;
import org.apache.commons.math3.linear.RealVector;

/**
 * http://stackoverflow.com/questions/10768142/verify-if-point-is-inside-a-cone-in-3d-space
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class ConeComputations {

    //Empty Constructor
    private ConeComputations(){
    }
    
    /**
     * http://stackoverflow.com/questions/10768142/verify-if-point-is-inside-a-cone-in-3d-space
     * <p>
     * @param x coordinates of point to be tested
     * @param coneEquation coordinates of cone
     * <p>
     * @return
     */
    public static boolean isLyingInCone(RealVector x,
            ConeEquation coneEquation) {

        // This is for our convenience
        Angle halfAperture = coneEquation.getAperture().divide(2.0);

        // Vector pointing to X point from apex
        RealVector apexToXVect = coneEquation.getT().subtract(x);

        // Vector pointing from apex to circle-center point.
        RealVector axisVect = coneEquation.getT().subtract(coneEquation.getB());

        /**
         * X is lying in cone only if it's lying in infinite version of its cone
         * -- that is, not limited by "round basement". We'll use dotProd() to
         * determine angle between apexToXVect and axis.
         */
        double m = apexToXVect.dotProduct(axisVect) / apexToXVect.getNorm() / axisVect.getNorm();

        /**
         * We can safely compare cos() of angles between vectors instead of bare
         * angles.
         */
        boolean isInInfiniteCone = m > halfAperture.cos();

        if (!isInInfiniteCone) {
            return false;
        }

        /**
         * X is contained in cone only if projection of apexToXVect to axis is
         * shorter than axis. We'll use dotProd() to figure projection length.
         */
        double n = apexToXVect.dotProduct(axisVect) / axisVect.getNorm();
        boolean isUnderRoundCap = n < axisVect.getNorm();

        return isUnderRoundCap;
    }

}
