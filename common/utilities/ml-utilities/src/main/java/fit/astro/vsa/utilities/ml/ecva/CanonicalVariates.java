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
package fit.astro.vsa.utilities.ml.ecva;

import java.util.Map;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class CanonicalVariates {

    private final Map<Integer, RealVector> mapOfPatterns;
    private final Map<Integer, String> mapOfClasses;

    private final Map<Integer, RealVector> canonicalVariates;
    private final RealVector mean;
    private final RealVector std;
    private final RealMatrix canonicalWeights;

    public CanonicalVariates(Map<Integer, RealVector> mapOfPatterns,
            Map<Integer, String> mapOfClasses,
            Map<Integer, RealVector> canonicalVariates,
            RealVector mean, RealVector std,
            RealMatrix canonicalWeights) {
        this.mapOfPatterns = mapOfPatterns;
        this.mapOfClasses = mapOfClasses;
        this.canonicalVariates = canonicalVariates;
        this.mean = mean;
        this.std = std;
        this.canonicalWeights = canonicalWeights;
    }

    /**
     * Apply the ECVA to an input vector
     *
     * @param input
     * @return
     */
    public RealVector applyECVA(RealVector input) {

        RealVector meanSub = input.subtract(mean);
        return canonicalWeights.transpose().operate(meanSub);

    }

    /**
     * @return the canonicalVariates
     */
    public Map<Integer, RealVector> getCanonicalVariates() {
        return canonicalVariates;
    }

    /**
     * @return the mapOfPatterns
     */
    public Map<Integer, RealVector> getMapOfPatterns() {
        return mapOfPatterns;
    }

    /**
     * @return the mapOfClasses
     */
    public Map<Integer, String> getMapOfClasses() {
        return mapOfClasses;
    }

    /**
     * @return the mean
     */
    public RealVector getMean() {
        return mean;
    }

    /**
     * @return the std
     */
    public RealVector getStd() {
        return std;
    }

    /**
     * @return the canonicalWeights
     */
    public RealMatrix getCanonicalWeights() {
        return canonicalWeights;
    }

}
