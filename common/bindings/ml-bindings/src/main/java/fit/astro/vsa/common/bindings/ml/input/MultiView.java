/*
 * Copyright (C) 2018 Kyle Johnston <kyjohnst2000@my.fit.edu>
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
package fit.astro.vsa.common.bindings.ml.input;

import java.util.Map;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class MultiView extends SideData {

    private Map<Integer, Map<String, RealVector>> mapOfVectorPatterns;
    private Map<Integer, Map<String, RealMatrix>> mapOfMatrixPatterns;

    /**
     *
     * @param subjectDescription What we are classifying
     * @param listOfVectorPatterns can be empty, list of vector variates
     * @param listOfMatrixPatterns can be empty, list of matrix variates
     * @param setOfClasses The class labels
     */
    public MultiView(String subjectDescription,
            Map<Integer, Map<String, RealVector>> listOfVectorPatterns,
            Map<Integer, Map<String, RealMatrix>> listOfMatrixPatterns,
            Map<Integer, String> setOfClasses) {
        super(subjectDescription, setOfClasses);
        this.mapOfVectorPatterns = listOfVectorPatterns;
        this.mapOfMatrixPatterns = listOfMatrixPatterns;
    }

    public Map<Integer, Map<String, RealVector>> getMapOfVectorPatterns() {
        return mapOfVectorPatterns;
    }

    public void setMapOfVectorPatterns(Map<Integer, Map<String, RealVector>> mapOfVectorPatterns) {
        this.mapOfVectorPatterns = mapOfVectorPatterns;
    }

    public Map<Integer, Map<String, RealMatrix>> getMapOfMatrixPatterns() {
        return mapOfMatrixPatterns;
    }

    public void setMapOfMatrixPatterns(Map<Integer, Map<String, RealMatrix>> mapOfMatrixPatterns) {
        this.mapOfMatrixPatterns = mapOfMatrixPatterns;
    }

}
