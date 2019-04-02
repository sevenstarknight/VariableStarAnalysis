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
package fit.astro.vsa.common.bindings.math;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class Prediction {

    private final double yEst;
    private final double ciUpper;
    private final double ciLower;

    /**
     *
     * @param yEst
     * @param var
     */
    public Prediction(double yEst, double var) {
        this.yEst = yEst;
        this.ciUpper = yEst + var;
        this.ciLower = yEst - var;
    }

    /**
     * @return the yEst
     */
    public double getyEst() {
        return yEst;
    }

    /**
     * @return the ciUpper
     */
    public double getCiUpper() {
        return ciUpper;
    }

    /**
     * @return the ciLower
     */
    public double getCiLower() {
        return ciLower;
    }
}
