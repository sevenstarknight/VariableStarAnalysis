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
package fit.astro.vsa.common.utilities.math.handling.sigfig;

import java.math.BigDecimal;
import java.math.MathContext;
import org.apache.commons.math3.analysis.UnivariateFunction;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class SignificantDigitsErrorFunction implements UnivariateFunction {

    private final int numberOfDigits;

    /**
     *
     * @param numberOfDigits
     */
    public SignificantDigitsErrorFunction(int numberOfDigits) {
        this.numberOfDigits = numberOfDigits;
    }

    /**
     * Set Significant Digits to 2
     *
     * @param d
     * @return
     */
    @Override
    public double value(double d) {
        BigDecimal bd = BigDecimal.valueOf(d);
        bd = bd.round(new MathContext(numberOfDigits));
        return bd.doubleValue();
    }

}
