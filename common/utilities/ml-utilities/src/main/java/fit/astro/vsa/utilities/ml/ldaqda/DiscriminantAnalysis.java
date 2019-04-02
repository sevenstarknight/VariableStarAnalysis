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
package fit.astro.vsa.utilities.ml.ldaqda;

import fit.astro.vsa.common.bindings.ml.DiscrimantAnalysisMethod;
import java.util.Map;
import fit.astro.vsa.common.bindings.ml.DiscrimantAnalysisResult;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author SevenStarKnight
 */
public class DiscriminantAnalysis {

    private final RealVector meanX;
    private final RealVector stdX;

    private final DiscrimantAnalysisMethod analysisMethod;
    private final Map<String, DiscrimantAnalysisResult> params;
    private final Map<String, Integer> uniqueLabelCount;

    public DiscriminantAnalysis(RealVector meanX, RealVector stdX, DiscrimantAnalysisMethod analysisMethod, Map<String, DiscrimantAnalysisResult> params, Map<String, Integer> uniqueLabelCount) {
        this.meanX = meanX;
        this.stdX = stdX;
        this.analysisMethod = analysisMethod;
        this.params = params;
        this.uniqueLabelCount = uniqueLabelCount;
    }

    public RealVector getMeanX() {
        return meanX;
    }

    public RealVector getStdX() {
        return stdX;
    }

    /**
     * @return the analysisMethod
     */
    public DiscrimantAnalysisMethod getAnalysisMethod() {
        return analysisMethod;
    }

    /**
     * @return the params
     */
    public Map<String, DiscrimantAnalysisResult> getParams() {
        return params;
    }

    public Map<String, Integer> getUniqueLabelCount() {
        return uniqueLabelCount;
    }

}
