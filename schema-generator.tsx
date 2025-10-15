

import { GeneratedContent, SiteInfo, ExpandedGeoTargeting, Reviewer } from './index.tsx';

export type WpConfig = {
    url: string;
    username: string;
};

// =================================================================
// 💎 HIGH-QUALITY SCHEMA.ORG MARKUP GENERATOR
// =================================================================
// This module creates SEO-optimized JSON-LD schema markup.
// It follows Google's latest guidelines to improve search visibility,
// enhance SERP rankings, and increase eligibility for rich snippets.
// =================================================================


// --- FALLBACKS: Used if no specific info is provided ---
const ORGANIZATION_NAME = "Your Company Name";
const DEFAULT_AUTHOR_NAME = "Expert Author";
// --- END FALLBACKS ---


/**
 * Creates a 'Person' schema object.
 * @param person Object containing name, URL, credentials, and social links.
 * @returns A Person schema object.
 */
function createPersonSchema(person: Partial<Reviewer> & { url?: string }) {
    return {
        "@type": "Person",
        "name": person.name || DEFAULT_AUTHOR_NAME,
        "url": person.url || undefined,
        "sameAs": person.sameAs && person.sameAs.length > 0 ? person.sameAs : undefined,
        "jobTitle": person.credentials || undefined,
        "image": person.imageUrl || undefined,
        "description": person.bio || undefined,
    };
}


/**
 * Creates an 'Organization' schema object, used for the publisher property.
 * @param siteInfo Object containing organization name, URL, logo, and social links.
 * @returns An Organization schema object.
 */
function createOrganizationSchema(siteInfo: SiteInfo) {
    return {
        "@type": "Organization",
        "name": siteInfo.orgName || ORGANIZATION_NAME,
        "url": siteInfo.orgUrl,
        "logo": siteInfo.logoUrl ? {
            "@type": "ImageObject",
            "url": siteInfo.logoUrl,
        } : undefined,
        "sameAs": siteInfo.orgSameAs && siteInfo.orgSameAs.length > 0 ? siteInfo.orgSameAs : undefined,
    };
}

/**
 * Creates a 'LocalBusiness' schema object for geo-targeted content.
 * @param siteInfo The organization's base information.
 * @param geoTargeting The detailed location information.
 * @returns A LocalBusiness schema object.
 */
function createLocalBusinessSchema(siteInfo: SiteInfo, geoTargeting: ExpandedGeoTargeting) {
    return {
        "@type": "LocalBusiness",
        "name": siteInfo.orgName || ORGANIZATION_NAME,
        "url": siteInfo.orgUrl,
        "address": {
            "@type": "PostalAddress",
            "addressLocality": geoTargeting.location,
            "addressRegion": geoTargeting.region,
            "postalCode": geoTargeting.postalCode,
            "addressCountry": geoTargeting.country,
        },
    };
}


/**
 * Creates the core 'Article' schema.
 * @param content The fully generated content object.
 * @param wpConfig The WordPress configuration containing the site URL.
 * @param orgSchema The generated Organization schema for the publisher.
 * @param authorSchema The generated Person schema for the author.
 * @param reviewerSchemas An array of Person schemas for the reviewers.
 * @returns An Article schema object.
 */
function createArticleSchema(content: GeneratedContent, wpConfig: WpConfig, orgSchema: object, authorSchema: object, reviewerSchemas: object[]) {
    const today = new Date().toISOString();
    return {
        "@type": "Article",
        "headline": content.title,
        "description": content.metaDescription,
        "image": content.imageDetails.map(img => img.generatedImageSrc).filter(Boolean),
        "datePublished": today,
        "dateModified": today,
        "author": authorSchema,
        "publisher": orgSchema,
        "reviewedBy": reviewerSchemas.length > 0 ? reviewerSchemas : undefined,
        "mainEntityOfPage": {
            "@type": "WebPage",
            "@id": `${wpConfig.url.replace(/\/+$/, '')}/${content.slug}`,
        },
        "keywords": content.semanticKeywords && content.semanticKeywords.length > 0 ? content.semanticKeywords.join(', ') : undefined,
    };
}

/**
 * Creates 'FAQPage' schema from structured data.
 * This is the preferred method as it's more reliable than HTML parsing.
 * @param faqData An array of question/answer objects.
 * @returns An FAQPage schema object, or null if no valid FAQs are provided.
 */
function createFaqSchema(faqData: { question: string, answer: string }[]) {
    if (!faqData || faqData.length === 0) {
        return null;
    }
    
    const mainEntity = faqData
        .filter(faq => faq.question && faq.answer)
        .map(faq => ({
            "@type": "Question",
            "name": faq.question,
            "acceptedAnswer": {
                "@type": "Answer",
                "text": faq.answer,
            },
        }));

    if (mainEntity.length === 0) return null;

    return {
        "@type": "FAQPage",
        "mainEntity": mainEntity,
    };
}

/**
 * [Fallback] Creates 'FAQPage' schema by parsing questions and answers from the final HTML content.
 * This version is improved to be more robust.
 * @param content The fully generated content object.
 * @returns An FAQPage schema object, or null if no valid FAQs are found.
 */
function createFaqSchemaFromHtml(content: GeneratedContent) {
    const mainEntity = [];
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = content.content;
    
    const headings = tempDiv.querySelectorAll('h2, h3');
    
    for (const heading of headings) {
        const questionText = heading.textContent?.trim();
        if (questionText && questionText.endsWith('?') && questionText.split(' ').length > 3) {
            let nextElement = heading.nextElementSibling;
            let answerText = '';
            let elementsInAnswer = 0;
            
            while (nextElement && !['H2', 'H3', 'H4'].includes(nextElement.tagName) && elementsInAnswer < 3) {
                if (nextElement.tagName === 'P' || nextElement.tagName === 'UL' || nextElement.tagName === 'OL') {
                    answerText += nextElement.textContent + ' ';
                    elementsInAnswer++;
                }
                nextElement = nextElement.nextElementSibling;
            }
            
            if (answerText) {
                mainEntity.push({
                    "@type": "Question",
                    "name": questionText,
                    "acceptedAnswer": {
                        "@type": "Answer",
// FIX: Complete the truncated function to resolve syntax errors.
                        "text": answerText.trim(),
                    },
                });
            }
        }
    }

    if (mainEntity.length === 0) return null;

    return {
        "@type": "FAQPage",
        "mainEntity": mainEntity,
    };
}

// ADD: Implement and export `generateFullSchema` and `generateSchemaMarkup` to resolve import errors in index.tsx.
/**
 * Generates a comprehensive JSON-LD schema graph.
 * This function orchestrates the creation of multiple linked schema types
 * (Article, Organization, Person, FAQPage, etc.) into a cohesive structure.
 * @param content The generated content object.
 * @param wpConfig WordPress configuration.
 * @param siteInfo Site-wide E-E-A-T and branding information.
 * @param faqData Structured FAQ data for schema generation.
 * @param geoTargeting Geo-targeting information for LocalBusiness schema.
 * @returns A JSON-LD schema object with a @graph of different schema types.
 */
export function generateFullSchema(
    content: GeneratedContent,
    wpConfig: WpConfig,
    siteInfo: SiteInfo,
    faqData: { question: string, answer: string }[],
    geoTargeting: ExpandedGeoTargeting
): object {
    const orgSchema = createOrganizationSchema(siteInfo);
    
    const authorSchema = createPersonSchema({
        name: siteInfo.authorName,
        url: siteInfo.authorUrl,
        sameAs: siteInfo.authorSameAs,
        credentials: siteInfo.authorCredentials,
    });
    
    const reviewerSchemas = (siteInfo.expertReviewers || [])
        .filter(r => r && r.name)
        .map(reviewer => createPersonSchema(reviewer));
    
    if (siteInfo.factChecker && siteInfo.factChecker.name) {
        reviewerSchemas.push(createPersonSchema(siteInfo.factChecker));
    }

    const articleSchema = createArticleSchema(content, wpConfig, orgSchema, authorSchema, reviewerSchemas);
    
    const faqSchema = createFaqSchema(faqData);

    const graph: object[] = [articleSchema];
    if (faqSchema) {
        graph.push(faqSchema);
    }

    // Add LocalBusiness schema if geo-targeting is enabled
    if (geoTargeting.enabled && geoTargeting.location) {
        const localBusinessSchema = createLocalBusinessSchema(siteInfo, geoTargeting);
        graph.push(localBusinessSchema);
    }
    
    return {
        "@context": "https://schema.org",
        "@graph": graph,
    };
}


/**
 * Wraps the generated JSON-LD schema object in the appropriate HTML for embedding in a page.
 * Uses a WordPress HTML block for compatibility.
 * @param schema The JSON-LD schema object.
 * @returns An HTML string containing the schema.
 */
export function generateSchemaMarkup(schema: object): string {
    try {
        const jsonLdString = JSON.stringify(schema, null, 2);
        // Using a WordPress HTML block is more robust for injection.
        return `
<!-- wp:html -->
<script type="application/ld+json">
${jsonLdString}
</script>
<!-- /wp:html -->
`;
    } catch (error) {
        console.error("Failed to stringify JSON-LD schema:", error);
        return '';
    }
}
