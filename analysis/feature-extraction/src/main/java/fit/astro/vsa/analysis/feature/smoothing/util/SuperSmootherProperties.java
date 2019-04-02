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
package fit.astro.vsa.analysis.feature.smoothing.util;

/**
 * Define properties and set default values for super smoother
 * @author Kyle.Johnston
 */
public class SuperSmootherProperties {

    private double span = 0;
    private double period = 0;
    private double alpha = 0;
    
    //Empty Constructor
    public SuperSmootherProperties() {
        
    }

    /**
     *
     * @param span
     * @param period
     * @param alpha
     */
    public SuperSmootherProperties(double span, 
            double period, double alpha) {
        
        if (span < 0 || span >= 1.0) {
            throw new ArithmeticException("Span must be between 0 and 1.0");
        }
        
        if (period < 0) {
            throw new ArithmeticException("Period can't be negative");
        }
        
        this.span = span;
        this.period = period;
        this.alpha = alpha;
    }
    
    


    /**
     * @return the span
     */
    public double getSpan() {
        return span;
    }

    /**
     * @param span the span to set
     */
    public void setSpan(double span) {
        this.span = span;
    }

    /**
     * @return the period
     */
    public double getPeriod() {
        return period;
    }

    /**
     * @param period the period to set
     */
    public void setPeriod(double period) {
        this.period = period;
    }

    /**
     * @return the alpha
     */
    public double getAlpha() {
        return alpha;
    }

    /**
     * @param alpha the alpha to set
     */
    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }



}
