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
package fit.astro.vsa.utilities.ml.pwc;

import fit.astro.vsa.common.bindings.math.kernel.KernelType;
import java.util.Map;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class PWC {
    
    
    private final Map<Integer, RealVector> setOfTrainingData;
    private final Map<Integer, String> setOfTrainingClasses;
    
    private final KernelType kernelType;
    private final double spread;
    private final String missedLabel;

    public PWC(Map<Integer, RealVector> setOfTrainingData, 
            Map<Integer, String> setOfTrainingClasses,
            KernelType kernelType, double spread, 
            String missedLabel) {
        
        this.setOfTrainingData = setOfTrainingData;
        this.setOfTrainingClasses = setOfTrainingClasses;
        
        this.kernelType = kernelType;
        this.spread = spread;
        this.missedLabel = missedLabel;
    }

    public Map<Integer, RealVector> getSetOfTrainingData() {
        return setOfTrainingData;
    }

    public Map<Integer, String> getSetOfTrainingClasses() {
        return setOfTrainingClasses;
    }

    public KernelType getKernelType() {
        return kernelType;
    }

    public double getSpread() {
        return spread;
    }

    public String getMissedLabel() {
        return missedLabel;
    }
    
    
}
