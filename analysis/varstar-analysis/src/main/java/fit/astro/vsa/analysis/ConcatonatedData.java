/*
 * Copyright (C) 2019 kjohnston
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
package fit.astro.vsa.analysis;

import java.io.Serializable;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author kjohnston
 */
public class ConcatonatedData implements Serializable {

    private static final long serialVersionUID = -6107478700048091667L;
    
    private final Map<Integer, RealVector> setOfPatterns_Training;
    private final Map<Integer, String> setOfClasses_Training;

    private final Map<Integer, RealVector> setOfPatterns_Testing;
    private final Map<Integer, String> setOfClasses_Testing;

    private final Map<Integer, List<Integer>> crossvalMap;

    public ConcatonatedData(Map<Integer, RealVector> setOfPatterns_Training, 
            Map<Integer, String> setOfClasses_Training,
            Map<Integer, RealVector> setOfPatterns_Testing, 
            Map<Integer, String> setOfClasses_Testing, 
            Map<Integer, List<Integer>> crossvalMap) {
        this.setOfPatterns_Training = setOfPatterns_Training;
        this.setOfClasses_Training = setOfClasses_Training;
        this.setOfPatterns_Testing = setOfPatterns_Testing;
        this.setOfClasses_Testing = setOfClasses_Testing;
        this.crossvalMap = crossvalMap;
    }

    public static long getSerialVersionUID() {
        return serialVersionUID;
    }

    public Map<Integer, RealVector> getSetOfPatterns_Training() {
        return setOfPatterns_Training;
    }

    public Map<Integer, String> getSetOfClasses_Training() {
        return setOfClasses_Training;
    }

    public Map<Integer, RealVector> getSetOfPatterns_Testing() {
        return setOfPatterns_Testing;
    }

    public Map<Integer, String> getSetOfClasses_Testing() {
        return setOfClasses_Testing;
    }

    public Map<Integer, List<Integer>> getCrossvalMap() {
        return crossvalMap;
    }

    
    
}
