/*
 * Copyright (C) 2018 Kyle Johnston
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
package fit.astro.vsa.common.bindings.analysis.astro;

import fit.astro.vsa.common.bindings.math.ml.input.MultiView;
import fit.astro.vsa.common.bindings.math.Real2DCurve;
import java.util.Map;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class VarStarDataset {
    
    private final String description;
    private final Map<Integer, Real2DCurve> timeD;
    private final Map<Integer, VarStarInformation> info;

    private MultiView multiViewSideData;
    
    public VarStarDataset(String description,
            Map<Integer, Real2DCurve> timeD, 
            Map<Integer, VarStarInformation> info) {
        this.description = description;
        this.timeD = timeD;
        this.info = info;
    }

    public String getDescription() {
        return description;
    }

    public Map<Integer, Real2DCurve> getTimeD() {
        return timeD;
    }

    public Map<Integer, VarStarInformation> getInfo() {
        return info;
    }
    
    
    
    // ===========================================================

    public MultiView getMultiViewSideData() {
        return multiViewSideData;
    }

    public void setMultiViewSideData(MultiView multiViewSideData) {
        this.multiViewSideData = multiViewSideData;
    }
    
    
    
    
}
